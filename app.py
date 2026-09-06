"""Personal YouTube Video Studio.

Run:
    streamlit run app.py

The app keeps API keys in Streamlit session state only. It does not upload them
or write them to disk. FFmpeg must be installed and available on PATH.
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests
import streamlit as st

APP_TITLE = "YouTube Video Studio"
USER_AGENT = "YouTubeVideoStudio/1.0 (+personal-use)"
REQUEST_TIMEOUT = 45


@dataclass
class Settings:
    llm_provider: str
    llm_api_key: str
    llm_model: str
    gemini_api_key: str
    gemini_model: str
    elevenlabs_key: str
    elevenlabs_voice_id: str
    media_provider: str
    media_api_key: str


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return value[:80] or "video"


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg topilmadi. FFmpeg o‘rnatib, PATH ga qo‘shing.")
    return path


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("Model javobida JSON topilmadi.")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("JSON obyekt bo‘lishi kerak.")
    return value


def request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    if not response.ok:
        detail = response.text[:1000]
        raise RuntimeError(f"API xatosi {response.status_code}: {detail}")
    return response.json()


def fetch_public_context(value: str) -> str:
    """Fetch a small amount of public page text when the user supplies a URL."""
    if not value.lower().startswith(("http://", "https://")):
        return value[:6000]
    try:
        response = requests.get(
            value,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        raw = response.text
        raw = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.I | re.S)
        raw = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.I | re.S)
        raw = re.sub(r"<[^>]+>", " ", raw)
        text = html.unescape(re.sub(r"\s+", " ", raw)).strip()
        return text[:12000]
    except requests.RequestException as exc:
        return f"URL mazmunini olishning iloji bo‘lmadi: {exc}. URLning o‘zini tahlil qiling."


def build_script_prompt(topic_or_url: str, fmt: str, source_context: str) -> str:
    if fmt == "shorts":
        dimensions = "9:16, 1080x1920"
        duration = "30–60 soniya"
    else:
        dimensions = "16:9, 1920x1080"
        duration = "2–8 daqiqa"
    return f"""
Siz YouTube uchun kuchli ssenariy va video-shot planner yaratuvchi AI arxitektorsiz.
Mavzu yoki kanal manzili:
{topic_or_url}

Ochiq manbadan olingan kontekst:
{source_context}

Video formati: {fmt} ({dimensions}), tavsiya etilgan davomiylik: {duration}.
Javobni faqat quyidagi JSON sxemasida qaytaring; markdown ishlatmang:
{{
  "title": "qisqa sarlavha",
  "hook": "birinchi 1-3 soniyadagi hook",
  "narration": "to‘liq voice-over matni",
  "description": "YouTube description",
  "tags": ["tag1", "tag2"],
  "shots": [
    {{
      "start": 0,
      "end": 4,
      "voiceover": "shu kadrda aytiladigan matn",
      "visual_query": "stock media qidiruv so‘zlari, ingliz tilida",
      "on_screen_text": "ekranga chiqadigan qisqa matn"
    }}
  ]
}}
Qoidalar:
- shots kamida 4 ta bo‘lsin.
- start/end soniyalarda bo‘lsin va uzluksiz ketma-ketlik hosil qilsin.
- Hook qiziqarli, lekin yolg‘on yoki chalg‘ituvchi bo‘lmasin.
- Mualliflik huquqini buzuvchi yoki xavfli kontentni tavsiya qilmang.
- Narration shots voiceover matnlarining mantiqiy yig‘indisi bo‘lsin.
""".strip()


def call_claude(settings: Settings, prompt: str) -> str:
    payload = {
        "model": settings.llm_model,
        "max_tokens": 5000,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = request_json(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": settings.llm_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
    )
    return "\n".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    )


def call_gemini(settings: Settings, prompt: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.llm_model}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "responseMimeType": "application/json"},
    }
    data = request_json("POST", url, params={"key": settings.llm_api_key}, json=payload)
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini javob qaytarmadi.")
    return candidates[0]["content"]["parts"][0]["text"]


ASSISTANT_INSTRUCTIONS = """
Siz YouTube Video Studio ichidagi shaxsiy Gemini yordamchisiz.
Foydalanuvchiga o‘zbek tilida, aniq va amaliy javob bering. Siz quyidagilarda yordam berasiz:
- YouTube mavzu, hook, ssenariy, shot timestamp va monetizatsiya xavflarini tushuntirish;
- hozirgi JSON ssenariydagi xatolarni topish va tuzatish;
- Streamlit, API, ElevenLabs, Pexels/Pixabay va FFmpeg bo‘yicha xatolarni izohlash;
- foydalanuvchi yozgan matnni imlo, mantiq va tuzilma jihatdan yaxshilash.

Agar foydalanuvchi ssenariyni tuzatishni so‘rasa, javob oxirida faqat bitta to‘liq JSON obyektini
`CORRECTED_JSON_START` va `CORRECTED_JSON_END` markerlari orasida qaytaring. JSON quyidagi
maydonlarni saqlasin: title, hook, narration, description, tags, shots. shots ichida start, end,
voiceover, visual_query va on_screen_text bo‘lsin. Tushuntirishni JSON tashqarisida yozing.
API kalitlarini hech qachon javobda takrorlamang.
""".strip()


def call_gemini_assistant(settings: Settings, user_message: str, script: dict[str, Any] | None) -> str:
    api_key = settings.gemini_api_key or (settings.llm_api_key if settings.llm_provider == "Gemini" else "")
    if not api_key:
        raise ValueError("Gemini Assistant API key kiritilmagan.")
    context = json.dumps(script, ensure_ascii=False, indent=2) if script else "Hozircha ssenariy yaratilmagan."
    prompt = f"""
{ASSISTANT_INSTRUCTIONS}

JORIY LOYIHA KONTEKSTI:
{context}

FOYDALANUVCHI SAVOLI:
{user_message}
""".strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.gemini_model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.35, "maxOutputTokens": 6000},
    }
    data = request_json("POST", url, params={"key": api_key}, json=payload)
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini Assistant javob qaytarmadi.")
    return candidates[0]["content"]["parts"][0].get("text", "").strip()


def extract_corrected_json(text: str) -> dict[str, Any] | None:
    start_marker = "CORRECTED_JSON_START"
    end_marker = "CORRECTED_JSON_END"
    if start_marker not in text or end_marker not in text:
        return None
    segment = text.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
    try:
        return parse_json_object(segment)
    except (ValueError, json.JSONDecodeError):
        return None


def assistant_panel(settings: Settings) -> None:
    st.subheader("🤖 Gemini yordamchi")
    st.caption("Savol bering, ssenariyni tekshirtiring yoki xatoni tushuntirishni so‘rang.")
    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = []
    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.button("Joriy ssenariyni tekshirish va xatolarni topish", use_container_width=True):
        if "script" not in st.session_state:
            st.warning("Avval ssenariy yarating.")
        else:
            question = (
                "Joriy ssenariyni to‘liq tekshir: timestamp uzluksizligini, narration va shot mosligini, "
                "hook kuchini, visual_query sifatini va monetizatsiya xavflarini ko‘rsat. "
                "Aniq tuzatishlar taklif qil va kerak bo‘lsa CORRECTED_JSON markerlari bilan to‘liq tuzatilgan JSON ber."
            )
            try:
                with st.spinner("Gemini ssenariyni tekshirmoqda..."):
                    answer = call_gemini_assistant(settings, question, st.session_state["script"])
                st.session_state["assistant_messages"].append({"role": "user", "content": question})
                st.session_state["assistant_messages"].append({"role": "assistant", "content": answer})
                corrected = extract_corrected_json(answer)
                if corrected:
                    st.session_state["assistant_corrected_json"] = corrected
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    user_question = st.chat_input("Gemini yordamchiga savol yozing...")
    if user_question:
        st.session_state["assistant_messages"].append({"role": "user", "content": user_question})
        try:
            with st.spinner("Gemini javob tayyorlamoqda..."):
                answer = call_gemini_assistant(settings, user_question, st.session_state.get("script"))
            st.session_state["assistant_messages"].append({"role": "assistant", "content": answer})
            corrected = extract_corrected_json(answer)
            if corrected:
                st.session_state["assistant_corrected_json"] = corrected
        except Exception as exc:
            st.session_state["assistant_messages"].append({"role": "assistant", "content": f"Xatolik: {exc}"})
        st.rerun()

    corrected = st.session_state.get("assistant_corrected_json")
    if corrected:
        st.warning("Gemini tuzatilgan JSON variantini taklif qildi.")
        if st.button("Tuzatilgan JSONni ssenariyga qo‘llash", use_container_width=True):
            st.session_state["script"] = corrected
            st.session_state["script_json"] = json.dumps(corrected, ensure_ascii=False, indent=2)
            st.session_state["approved"] = True
            st.session_state.pop("assistant_corrected_json", None)
            st.success("Tuzatilgan ssenariy qo‘llandi.")
            st.rerun()
def generate_script(settings: Settings, topic: str, fmt: str) -> dict[str, Any]:
    source_context = fetch_public_context(topic)
    prompt = build_script_prompt(topic, fmt, source_context)
    if settings.llm_provider == "Claude":
        raw = call_claude(settings, prompt)
    else:
        raw = call_gemini(settings, prompt)
    result = parse_json_object(raw)
    shots = result.get("shots")
    if not isinstance(shots, list) or len(shots) < 1:
        raise ValueError("Ssenariyda shots ro‘yxati topilmadi.")
    normalized = []
    for index, shot in enumerate(shots):
        try:
            start = float(shot.get("start", 0))
            end = float(shot.get("end", start + 4))
        except (TypeError, ValueError):
            start, end = index * 4, (index + 1) * 4
        normalized.append({
            "start": max(0, start),
            "end": max(start + 0.5, end),
            "voiceover": str(shot.get("voiceover", "")),
            "visual_query": str(shot.get("visual_query", "abstract cinematic background")),
            "on_screen_text": str(shot.get("on_screen_text", "")),
        })
    result["shots"] = normalized
    result.setdefault("narration", " ".join(s["voiceover"] for s in normalized))
    return result


def demo_script(topic: str) -> dict[str, Any]:
    return {
        "title": f"{topic[:45]}: 3 muhim g‘oya",
        "hook": "Siz bu mavzudagi eng muhim uchta nuqtani bilasizmi?",
        "narration": (
            f"Bugun {topic} haqida uchta muhim fikrni ko‘rib chiqamiz. "
            "Birinchisi — asosiy muammo va uning sababi. Ikkinchisi — amaliy yechim. "
            "Uchinchisi — bugunoq sinab ko‘rish mumkin bo‘lgan qadam."
        ),
        "description": "Shaxsiy demo video.",
        "tags": [topic, "education", "uzbek"],
        "shots": [
            {"start": 0, "end": 4, "voiceover": "Bugun bu mavzudagi uchta muhim fikrni ko‘ramiz.", "visual_query": topic, "on_screen_text": "3 MUHIM FIKR"},
            {"start": 4, "end": 9, "voiceover": "Birinchisi — asosiy muammo va uning sababi.", "visual_query": "problem solution concept", "on_screen_text": "1. MUAMMO"},
            {"start": 9, "end": 15, "voiceover": "Ikkinchisi — amaliy yechim.", "visual_query": "person planning solution", "on_screen_text": "2. YECHIM"},
            {"start": 15, "end": 22, "voiceover": "Uchinchisi — bugunoq sinab ko‘rish mumkin bo‘lgan qadam.", "visual_query": "success action hands technology", "on_screen_text": "3. QADAM"},
        ],
    }


def search_pexels(api_key: str, query: str, video: bool = True) -> dict[str, Any] | None:
    if video:
        data = request_json(
            "GET",
            "https://api.pexels.com/videos/search",
            headers={"Authorization": api_key},
            params={"query": query, "per_page": 10, "orientation": "portrait"},
        )
        for item in data.get("videos", []):
            files = sorted(item.get("video_files", []), key=lambda x: x.get("width", 0) * x.get("height", 0), reverse=True)
            for file in files:
                if file.get("link"):
                    return {"url": file["link"], "kind": "video"}
    else:
        data = request_json(
            "GET",
            "https://api.pexels.com/v1/search",
            headers={"Authorization": api_key},
            params={"query": query, "per_page": 10, "orientation": "portrait"},
        )
        for item in data.get("photos", []):
            url = item.get("src", {}).get("large2x") or item.get("src", {}).get("large")
            if url:
                return {"url": url, "kind": "image"}
    return None


def search_pixabay(api_key: str, query: str, video: bool = True) -> dict[str, Any] | None:
    if video:
        data = request_json(
            "GET",
            "https://pixabay.com/api/videos/",
            params={"key": api_key, "q": query, "per_page": 10, "safesearch": "true"},
        )
        for item in data.get("hits", []):
            videos = item.get("videos", {})
            for key in ("large", "medium", "small", "tiny"):
                if videos.get(key, {}).get("url"):
                    return {"url": videos[key]["url"], "kind": "video"}
    else:
        data = request_json(
            "GET",
            "https://pixabay.com/api/",
            params={"key": api_key, "q": query, "per_page": 10, "safesearch": "true", "image_type": "photo"},
        )
        for item in data.get("hits", []):
            if item.get("largeImageURL"):
                return {"url": item["largeImageURL"], "kind": "image"}
    return None


def download(url: str, target: Path) -> Path:
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return target


def get_media(settings: Settings, query: str, workdir: Path, index: int) -> tuple[Path, str]:
    providers = [(settings.media_provider, True), (settings.media_provider, False)]
    for provider, is_video in providers:
        try:
            result = search_pexels(settings.media_api_key, query, is_video) if provider == "Pexels" else search_pixabay(settings.media_api_key, query, is_video)
            if result:
                extension = ".mp4" if result["kind"] == "video" else ".jpg"
                return download(result["url"], workdir / f"media_{index}{extension}"), result["kind"]
        except Exception:
            continue
    raise RuntimeError(f"Media topilmadi: {query}")


def elevenlabs_tts(settings: Settings, text: str, output: Path) -> Path:
    data = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}",
        headers={"xi-api-key": settings.elevenlabs_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        json={"text": text, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 0.45, "similarity_boost": 0.8}},
        timeout=REQUEST_TIMEOUT,
    )
    if not data.ok:
        raise RuntimeError(f"ElevenLabs xatosi {data.status_code}: {data.text[:500]}")
    output.write_bytes(data.content)
    return output


def video_filter(width: int, height: int) -> str:
    return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,format=yuv420p"


def render_segment(media: Path, kind: str, output: Path, duration: float, width: int, height: int) -> None:
    ffmpeg = ffmpeg_path()
    vf = video_filter(width, height)
    if kind == "image":
        args = [ffmpeg, "-y", "-loop", "1", "-i", str(media), "-t", str(duration), "-vf", vf, "-r", "30", "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output)]
    else:
        args = [ffmpeg, "-y", "-stream_loop", "-1", "-i", str(media), "-t", str(duration), "-vf", vf, "-r", "30", "-an", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output)]
    run_command(args)


def concat_segments(segments: list[Path], output: Path) -> Path:
    ffmpeg = ffmpeg_path()
    list_file = output.parent / "concat.txt"
    list_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in segments), encoding="utf-8")
    run_command([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(output)])
    return output


def mux_audio(video: Path, audio: Path, output: Path) -> Path:
    ffmpeg = ffmpeg_path()
    filter_complex = "[1:a]afade=t=in:st=0:d=2[a]"
    run_command([
        ffmpeg, "-y", "-i", str(video), "-i", str(audio),
        "-filter_complex", filter_complex,
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", "-movflags", "+faststart",
        str(output),
    ])
    return output


def render_project(settings: Settings, script: dict[str, Any], fmt: str, progress: Callable[[int, str], None]) -> Path:
    width, height = (1080, 1920) if fmt == "shorts" else (1920, 1080)
    workdir = Path(tempfile.mkdtemp(prefix="ytstudio_"))
    try:
        shots = script["shots"]
        rendered: list[Path] = []
        for index, shot in enumerate(shots):
            progress(30 + int((index / max(1, len(shots))) * 30), f"Vizual kadr {index + 1}/{len(shots)} qidirilmoqda")
            media, kind = get_media(settings, shot["visual_query"], workdir, index)
            duration = max(0.7, float(shot["end"]) - float(shot["start"]))
            segment = workdir / f"segment_{index}.mp4"
            render_segment(media, kind, segment, duration, width, height)
            rendered.append(segment)
        progress(65, "Voice-over tayyorlanmoqda")
        audio = elevenlabs_tts(settings, script.get("narration", ""), workdir / "voiceover.mp3")
        progress(82, "FFmpeg segmentlarni birlashtirmoqda")
        silent = concat_segments(rendered, workdir / "silent.mp4")
        progress(92, "Audio fade-in va sinxronizatsiya")
        final = workdir / f"{safe_name(script.get('title', 'video'))}.mp4"
        mux_audio(silent, audio, final)
        progress(100, "Tayyor")
        persistent = Path("outputs")
        persistent.mkdir(exist_ok=True)
        destination = persistent / final.name
        shutil.copy2(final, destination)
        return destination
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise


def get_settings() -> Settings:
    with st.sidebar:
        st.header("Settings")
        st.caption("Kalitlar faqat joriy sessiya xotirasida saqlanadi.")
        llm_provider = st.selectbox("AI provider", ["Claude", "Gemini"])
        if llm_provider == "Claude":
            default_model = "claude-3-5-sonnet-latest"
        else:
            default_model = "gemini-2.0-flash"
        llm_model = st.text_input("AI model", value=default_model)
        llm_api_key = st.text_input("Claude/Gemini API key", type="password")
        st.divider()
        st.subheader("Gemini Assistant")
        gemini_api_key = st.text_input("Gemini Assistant API key", type="password")
        gemini_model = st.text_input("Gemini Assistant model", value="gemini-2.0-flash")
        elevenlabs_key = st.text_input("ElevenLabs API key", type="password")
        elevenlabs_voice_id = st.text_input("ElevenLabs Voice ID", value="21m00Tcm4TlvDq8ikWAM")
        media_provider = st.selectbox("Stock media provider", ["Pexels", "Pixabay"])
        media_api_key = st.text_input(f"{media_provider} API key", type="password")
        st.divider()
        st.markdown("**Kerakli dastur:** `ffmpeg` PATH ichida bo‘lishi kerak.")
    return Settings(llm_provider, llm_api_key, llm_model, gemini_api_key, gemini_model, elevenlabs_key, elevenlabs_voice_id, media_provider, media_api_key)


def validate_settings(settings: Settings, need_ai: bool = True) -> None:
    missing = []
    if need_ai and not settings.llm_api_key:
        missing.append("AI API key")
    if not settings.media_api_key:
        missing.append(f"{settings.media_provider} API key")
    if not settings.elevenlabs_key:
        missing.append("ElevenLabs API key")
    if missing:
        raise ValueError("Settings bo‘limida quyidagilarni kiriting: " + ", ".join(missing))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎬", layout="wide")
    st.title("🎬 YouTube Video Studio")
    st.caption("Shaxsiy foydalanish uchun: ssenariy → stock media → voice-over → watermark-free MP4")
    settings = get_settings()

    left, right = st.columns([1.35, 1])
    with left:
        topic = st.text_area("Mavzu yoki raqobatchi YouTube kanal havolasi", height=90, placeholder="Masalan: Sun’iy intellektning 2026-yildagi 5 ta amaliy foydasi")
        fmt_label = st.radio("Video formati", ["YouTube Shorts — 9:16", "Long Video — 16:9"], horizontal=True)
        fmt = "shorts" if fmt_label.startswith("YouTube") else "long"
        mode = st.radio("Ishlash rejimi", ["Full Auto", "Semi-Auto"], horizontal=True)
        use_demo = st.checkbox("API kalitlarisiz demo ssenariy ishlatish", value=False)
        generate = st.button("1. Ssenariy yaratish", type="primary", use_container_width=True)
    with right:
        st.info("Semi-Auto rejimida JSON ssenariyni tasdiqlash yoki o‘zgartirish mumkin. Full Auto rejimida tasdiqlash bosqichi o‘tkazib yuboriladi.")
        st.markdown("**Format:** " + ("1080×1920" if fmt == "shorts" else "1920×1080"))

    if generate:
        if not topic.strip():
            st.error("Mavzu yoki kanal havolasini kiriting.")
        else:
            try:
                if use_demo:
                    script = demo_script(topic)
                else:
                    validate_settings(settings, need_ai=True)
                    script = generate_script(settings, topic, fmt)
                st.session_state["script"] = script
                st.session_state["script_json"] = json.dumps(script, ensure_ascii=False, indent=2)
                st.session_state["approved"] = mode == "Full Auto"
                st.success("Ssenariy tayyor.")
            except Exception as exc:
                st.error(str(exc))

    if "script" in st.session_state:
        st.subheader("Ssenariy va shot planner")
        edited = st.text_area("JSON (Semi-Auto rejimida tahrirlashingiz mumkin)", value=st.session_state.get("script_json", ""), height=420)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ssenariyni tasdiqlash / yangilash", use_container_width=True):
                try:
                    st.session_state["script"] = parse_json_object(edited)
                    st.session_state["script_json"] = json.dumps(st.session_state["script"], ensure_ascii=False, indent=2)
                    st.session_state["approved"] = True
                    st.success("Ssenariy tasdiqlandi.")
                except Exception as exc:
                    st.error(f"JSON xatosi: {exc}")
        with col2:
            render = st.button("2. MP4 render qilish", use_container_width=True, disabled=not st.session_state.get("approved", False))
        if render:
            try:
                validate_settings(settings, need_ai=False)
                ffmpeg_path()
                bar = st.progress(0)
                status = st.empty()
                def progress(value: int, message: str) -> None:
                    bar.progress(min(100, max(0, value)))
                    status.write(f"{value}% — {message}")
                output = render_project(settings, st.session_state["script"], fmt, progress)
                st.session_state["output"] = str(output)
                st.success("Video tayyor.")
            except Exception as exc:
                st.error(str(exc))

    if st.session_state.get("output"):
        output = Path(st.session_state["output"])
        if output.exists():
            st.subheader("Natija")
            st.video(str(output))
            st.download_button("MP4 videoni yuklab olish", data=output.read_bytes(), file_name=output.name, mime="video/mp4")

    st.divider()
    assistant_panel(settings)


if __name__ == "__main__":
    main()
