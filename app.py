"""YouTube Video Studio Pro - multilingual AI video pipeline.

Streamlit + Gemini/Claude + ElevenLabs + Pexels/Pixabay + YouTube Data API + FFmpeg.
API keys live only in st.session_state and are never written to disk.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
import streamlit as st

import youtube_api
from i18n import LANGUAGES, tr

APP_TITLE = "YouTube Video Studio Pro"
USER_AGENT = "YouTubeVideoStudio/3.0 (+personal-use)"
REQUEST_TIMEOUT = 60
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_FALLBACKS = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-flash-latest"]
CONTENT_LANGUAGES = {
    "uz": {"uz": "O‘zbekcha", "ru": "Узбекский", "en": "Uzbek"},
    "ru": {"uz": "Ruscha", "ru": "Русский", "en": "Russian"},
    "en": {"uz": "Inglizcha", "ru": "Английский", "en": "English"},
}
CONTENT_LANGUAGE_PROMPT = {
    "uz": "Butun ssenariy, narration va description o‘zbek tilida bo‘lsin.",
    "ru": "Весь сценарий, озвучка и описание должны быть на русском языке.",
    "en": "The whole script, narration and description must be in English.",
}

THEME_CSS = """
<style>
:root { --ys-red:#ff2d55; --ys-card:#151926; --ys-line:#242a3d; }
.stApp { background: radial-gradient(1200px 600px at 15% -10%, #1d1330 0%, #0d0f16 55%); }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.1rem; max-width: 1180px; }
.ys-hero { background: linear-gradient(135deg, rgba(255,45,85,.16), rgba(120,80,255,.16));
  border:1px solid var(--ys-line); border-radius:22px; padding:26px 24px; margin-bottom:16px; }
.ys-hero h1 { font-size:2rem; font-weight:800; margin:0 0 6px; letter-spacing:-.5px; }
.ys-hero p { margin:0; color:#9aa4bf; font-size:.94rem; }
.ys-badge { display:inline-block; padding:4px 11px; border-radius:999px; font-size:.7rem; font-weight:700;
  letter-spacing:.6px; background:rgba(255,45,85,.16); color:#ff6b88;
  border:1px solid rgba(255,45,85,.35); margin-bottom:12px; }
.ys-card { background:var(--ys-card); border:1px solid var(--ys-line); border-radius:18px;
  padding:16px 18px 8px; margin-bottom:14px; }
.ys-card h3 { font-size:1rem; margin:0 0 8px; color:#e7ebf7; font-weight:700; }
.ys-steps { display:flex; gap:8px; flex-wrap:wrap; margin:2px 0 16px; }
.ys-step { flex:1 1 110px; background:var(--ys-card); border:1px solid var(--ys-line);
  border-radius:14px; padding:10px 12px; font-size:.76rem; color:#7c86a3; }
.ys-step b { display:block; font-size:.88rem; color:#c9d2e8; margin-top:2px; }
.ys-step.done { border-color:rgba(46,204,113,.5); background:rgba(46,204,113,.08); }
.ys-step.done b { color:#7ee2a8; }
.ys-step.active { border-color:rgba(255,45,85,.55); background:rgba(255,45,85,.1); }
.ys-step.active b { color:#ff8aa2; }
.stButton > button { border-radius:12px; font-weight:700; border:1px solid var(--ys-line);
  background:#1b2133; color:#e7ebf7; padding:.6rem 1rem; }
.stButton > button:hover { border-color:var(--ys-red); color:#fff; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#ff2d55,#ff5f7e); border:0; color:#fff; }
div[data-testid="stTextInput"] input, div[data-testid="stTextArea"] textarea {
  background:#101522 !important; border-radius:12px !important;
  border:1px solid var(--ys-line) !important; color:#e7ebf7 !important; }
div[data-testid="stSidebar"] { background:#0b0e17; border-right:1px solid var(--ys-line); }
div[data-testid="stMetric"] { background:var(--ys-card); border:1px solid var(--ys-line);
  border-radius:14px; padding:12px 14px; }
.stTabs [data-baseweb="tab-list"] { gap:6px; flex-wrap:wrap; }
.stTabs [data-baseweb="tab"] { background:#141827; border:1px solid var(--ys-line);
  border-radius:12px 12px 0 0; padding:8px 14px; font-weight:600; }
.stTabs [aria-selected="true"] { background:rgba(255,45,85,.14); border-color:rgba(255,45,85,.4); }
.ys-shot { background:#111624; border:1px solid var(--ys-line); border-left:3px solid var(--ys-red);
  border-radius:12px; padding:10px 13px; margin-bottom:9px; }
.ys-shot .t { color:#ff8aa2; font-size:.74rem; font-weight:700; }
.ys-shot .v { color:#dbe2f2; font-size:.9rem; margin:3px 0; }
.ys-shot .q { color:#7c86a3; font-size:.77rem; }
.ys-log { background:#0f1421; border:1px solid var(--ys-line); border-radius:14px; padding:12px 14px; }
.ys-row { display:flex; justify-content:space-between; gap:10px; font-size:.85rem;
  padding:5px 0; border-bottom:1px dashed rgba(255,255,255,.06); }
.ys-row:last-child { border-bottom:0; }
.ys-row .n { color:#c9d2e8; }
.ys-row .s { color:#7c86a3; font-size:.8rem; white-space:nowrap; }
.ys-row.run .n { color:#ff8aa2; font-weight:700; }
.ys-row.ok .n { color:#7ee2a8; }
.ys-row.err .n { color:#ff7b7b; }
.ys-tip { color:#7c86a3; font-size:.82rem; }
.ys-pill { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.72rem;
  font-weight:700; margin-left:6px; }
.ys-pill.on { background:rgba(46,204,113,.14); color:#7ee2a8; border:1px solid rgba(46,204,113,.4); }
.ys-pill.off { background:rgba(255,255,255,.05); color:#8b95b0; border:1px solid var(--ys-line); }
</style>
"""


@dataclass
class Settings:
    ui_lang: str = "uz"
    content_lang: str = "uz"
    llm_provider: str = "Gemini"
    llm_api_key: str = ""
    llm_model: str = DEFAULT_GEMINI_MODEL
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    elevenlabs_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    media_provider: str = "Pexels"
    media_api_key: str = ""
    youtube_api_key: str = ""


@dataclass
class StageTracker:
    """Renders a detailed, live pipeline indicator."""

    labels: list[tuple[str, str]]
    lang: str
    placeholder: Any = None
    bar: Any = None
    state: dict[str, str] = field(default_factory=dict)
    detail: dict[str, str] = field(default_factory=dict)
    started: dict[str, float] = field(default_factory=dict)
    elapsed: dict[str, float] = field(default_factory=dict)

    def mount(self) -> None:
        self.bar = st.progress(0)
        self.placeholder = st.empty()
        self.draw(0)

    def draw(self, percent: int) -> None:
        rows = []
        for key, label in self.labels:
            status = self.state.get(key, "wait")
            css = {"run": "ys-row run", "ok": "ys-row ok", "err": "ys-row err"}.get(status, "ys-row")
            icon = {"run": "*", "ok": "OK", "err": "!"}.get(status, "-")
            right = self.detail.get(key, "")
            if key in self.elapsed:
                right = f"{right} · {self.elapsed[key]:.1f}s".strip(" ·")
            rows.append(
                f"<div class='{css}'><span class='n'>{icon} {html.escape(label)}</span>"
                f"<span class='s'>{html.escape(right)}</span></div>"
            )
        self.placeholder.markdown(f"<div class='ys-log'>{''.join(rows)}</div>", unsafe_allow_html=True)
        if self.bar is not None:
            self.bar.progress(min(100, max(0, percent)))

    def start(self, key: str, percent: int, detail: str = "") -> None:
        self.state[key] = "run"
        self.detail[key] = detail
        self.started[key] = time.time()
        self.draw(percent)

    def update(self, key: str, percent: int, detail: str) -> None:
        self.detail[key] = detail
        self.draw(percent)

    def done(self, key: str, percent: int, detail: str = "") -> None:
        self.state[key] = "ok"
        if detail:
            self.detail[key] = detail
        if key in self.started:
            self.elapsed[key] = time.time() - self.started[key]
        self.draw(percent)

    def fail(self, key: str, detail: str) -> None:
        self.state[key] = "err"
        self.detail[key] = detail[:80]
        self.draw(100)


def safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return value[:80] or "video"


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg not found on the server.")
    return path


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True)


def parse_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("JSON not found in model output.")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("JSON object expected.")
    return value


def request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    if not response.ok:
        raise RuntimeError(f"API {response.status_code}: {response.text[:600]}")
    return response.json()


def friendly_error(message: str, lang: str) -> str:
    table = {
        "model": {
            "uz": "Tanlangan model endi mavjud emas. Sozlamalarda modelni yangilang.",
            "ru": "Выбранная модель больше недоступна. Обновите модель в настройках.",
            "en": "The selected model is no longer available. Update it in settings.",
        },
        "busy": {
            "uz": "Server hozir band (503). 1–3 daqiqadan keyin qayta urinib ko‘ring.",
            "ru": "Сервер занят (503). Повторите через 1–3 минуты.",
            "en": "The service is busy (503). Try again in 1–3 minutes.",
        },
        "limit": {
            "uz": "API limiti tugadi (429). Bir oz kutib qayta urinib ko‘ring.",
            "ru": "Лимит API исчерпан (429). Подождите и повторите.",
            "en": "API quota exceeded (429). Wait and retry.",
        },
        "auth": {
            "uz": "API kalit qabul qilinmadi. Sozlamalardagi kalitni tekshiring.",
            "ru": "API ключ не принят. Проверьте ключ в настройках.",
            "en": "API key was rejected. Check the key in settings.",
        },
    }
    if "no longer available" in message or "NOT_FOUND" in message:
        return table["model"][lang]
    if "503" in message or "UNAVAILABLE" in message:
        return table["busy"][lang]
    if "429" in message:
        return table["limit"][lang]
    if any(code in message for code in ("401", "403")) or "API_KEY" in message.upper():
        return table["auth"][lang]
    return message


def fetch_public_context(value: str) -> str:
    if not value.lower().startswith(("http://", "https://")):
        return value[:6000]
    try:
        response = requests.get(value, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        raw = re.sub(r"<script[^>]*>.*?</script>", " ", response.text, flags=re.I | re.S)
        raw = re.sub(r"<style[^>]*>.*?</style>", " ", raw, flags=re.I | re.S)
        raw = re.sub(r"<[^>]+>", " ", raw)
        return html.unescape(re.sub(r"\s+", " ", raw)).strip()[:12000]
    except requests.RequestException as exc:
        return f"URL content unavailable: {exc}"


def build_script_prompt(topic: str, fmt: str, context: str, content_lang: str, template_block: str) -> str:
    dims = "9:16, 1080x1920" if fmt == "shorts" else "16:9, 1920x1080"
    duration = "30-60s" if fmt == "shorts" else "2-8 min"
    return f"""
You are a YouTube script and shot-planner architect.
{CONTENT_LANGUAGE_PROMPT[content_lang]}

TOPIC OR CHANNEL:
{topic}

PUBLIC CONTEXT:
{context}

{template_block}

Format: {fmt} ({dims}), target duration: {duration}.
Return ONLY this JSON schema, no markdown:
{{
  "title": "...", "hook": "...", "narration": "...", "description": "...",
  "tags": ["..."],
  "shots": [{{"start":0,"end":4,"voiceover":"...","visual_query":"english stock search words","on_screen_text":"..."}}]
}}
Rules: at least 4 shots; start/end in seconds and continuous; no misleading hook;
no copyright-infringing or unsafe advice; narration must equal the ordered shot voiceovers;
visual_query must always be in English for stock search.
""".strip()


def call_claude(settings: Settings, prompt: str) -> str:
    data = request_json(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": settings.llm_api_key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": settings.llm_model, "max_tokens": 5000, "temperature": 0.7,
              "messages": [{"role": "user", "content": prompt}]},
    )
    return "\n".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def gemini_generate(api_key: str, model: str, prompt: str, json_mode: bool, temperature: float) -> str:
    models = [model] + [m for m in GEMINI_FALLBACKS if m != model]
    config: dict[str, Any] = {"temperature": temperature}
    if json_mode:
        config["responseMimeType"] = "application/json"
    last_error = ""
    for candidate in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate}:generateContent"
        try:
            data = request_json("POST", url, params={"key": api_key},
                                json={"contents": [{"parts": [{"text": prompt}]}],
                                      "generationConfig": config})
        except RuntimeError as exc:
            last_error = str(exc)
            if any(code in last_error for code in ("404", "429", "500", "503")):
                continue
            raise
        candidates = data.get("candidates", [])
        if not candidates:
            last_error = "Empty Gemini response."
            continue
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if text:
            st.session_state["active_gemini_model"] = candidate
            return text
        last_error = "Gemini returned no text."
    raise RuntimeError(last_error or "Gemini did not answer.")


def normalize_script(result: dict[str, Any]) -> dict[str, Any]:
    shots = result.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("Script has no shots list.")
    normalized = []
    for index, shot in enumerate(shots):
        try:
            start = float(shot.get("start", index * 4))
            end = float(shot.get("end", start + 4))
        except (TypeError, ValueError):
            start, end = index * 4.0, index * 4.0 + 4
        normalized.append({
            "start": max(0.0, start),
            "end": max(start + 0.5, end),
            "voiceover": str(shot.get("voiceover", "")),
            "visual_query": str(shot.get("visual_query", "cinematic abstract background")),
            "on_screen_text": str(shot.get("on_screen_text", "")),
        })
    result["shots"] = normalized
    result.setdefault("narration", " ".join(s["voiceover"] for s in normalized))
    return result


def generate_script(settings: Settings, topic: str, fmt: str, template_block: str,
                    tracker: StageTracker) -> dict[str, Any]:
    tracker.start("analyze", 6)
    context = fetch_public_context(topic)
    tracker.done("analyze", 14, f"{len(context)} chars")
    tracker.start("script", 20, settings.llm_model)
    prompt = build_script_prompt(topic, fmt, context, settings.content_lang, template_block)
    raw = call_claude(settings, prompt) if settings.llm_provider == "Claude" else gemini_generate(
        settings.llm_api_key, settings.llm_model, prompt, True, 0.7)
    script = normalize_script(parse_json_object(raw))
    tracker.done("script", 100, f"{len(script['shots'])} shots")
    return script


def demo_script(topic: str, lang: str) -> dict[str, Any]:
    intro, labels = {
        "uz": (f"Bugun {topic} haqida uchta muhim fikrni ko‘ramiz.", ["MUAMMO", "YECHIM", "QADAM"]),
        "ru": (f"Сегодня разберём три главные мысли о {topic}.", ["ПРОБЛЕМА", "РЕШЕНИЕ", "ШАГ"]),
        "en": (f"Today we cover three key ideas about {topic}.", ["PROBLEM", "SOLUTION", "STEP"]),
    }[lang]
    return normalize_script({
        "title": f"{topic[:45]} - 3 ideas",
        "hook": intro,
        "description": "Personal demo video.",
        "tags": [topic, "education"],
        "shots": [
            {"start": 0, "end": 4, "voiceover": intro, "visual_query": topic, "on_screen_text": "3 IDEAS"},
            {"start": 4, "end": 9, "voiceover": labels[0], "visual_query": "problem concept dark",
             "on_screen_text": f"1. {labels[0]}"},
            {"start": 9, "end": 15, "voiceover": labels[1], "visual_query": "person planning solution",
             "on_screen_text": f"2. {labels[1]}"},
            {"start": 15, "end": 22, "voiceover": labels[2], "visual_query": "success hands technology",
             "on_screen_text": f"3. {labels[2]}"},
        ],
    })


def search_pexels(api_key: str, query: str, video: bool, portrait: bool) -> dict[str, Any] | None:
    orientation = "portrait" if portrait else "landscape"
    if video:
        data = request_json("GET", "https://api.pexels.com/videos/search",
                            headers={"Authorization": api_key},
                            params={"query": query, "per_page": 10, "orientation": orientation})
        for item in data.get("videos", []):
            files = sorted(item.get("video_files", []),
                           key=lambda x: (x.get("width") or 0) * (x.get("height") or 0), reverse=True)
            for file in files:
                if file.get("link"):
                    return {"url": file["link"], "kind": "video"}
        return None
    data = request_json("GET", "https://api.pexels.com/v1/search",
                        headers={"Authorization": api_key},
                        params={"query": query, "per_page": 10, "orientation": orientation})
    for item in data.get("photos", []):
        url = item.get("src", {}).get("large2x") or item.get("src", {}).get("large")
        if url:
            return {"url": url, "kind": "image"}
    return None


def search_pixabay(api_key: str, query: str, video: bool, portrait: bool) -> dict[str, Any] | None:
    if video:
        data = request_json("GET", "https://pixabay.com/api/videos/",
                            params={"key": api_key, "q": query, "per_page": 10, "safesearch": "true"})
        for item in data.get("hits", []):
            videos = item.get("videos", {})
            for key in ("large", "medium", "small", "tiny"):
                if videos.get(key, {}).get("url"):
                    return {"url": videos[key]["url"], "kind": "video"}
        return None
    data = request_json("GET", "https://pixabay.com/api/",
                        params={"key": api_key, "q": query, "per_page": 10, "safesearch": "true",
                                "image_type": "photo",
                                "orientation": "vertical" if portrait else "horizontal"})
    for item in data.get("hits", []):
        if item.get("largeImageURL"):
            return {"url": item["largeImageURL"], "kind": "image"}
    return None


def download(url: str, target: Path) -> Path:
    with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True,
                      timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return target


def get_media(settings: Settings, query: str, workdir: Path, index: int,
              portrait: bool) -> tuple[Path, str]:
    finder = search_pexels if settings.media_provider == "Pexels" else search_pixabay
    for attempt in (query, " ".join(query.split()[:2]) or query, "abstract cinematic background"):
        for is_video in (True, False):
            try:
                result = finder(settings.media_api_key, attempt, is_video, portrait)
            except Exception:
                result = None
            if result:
                extension = ".mp4" if result["kind"] == "video" else ".jpg"
                return download(result["url"], workdir / f"media_{index}{extension}"), result["kind"]
    raise RuntimeError(f"No media found for: {query}")


def elevenlabs_tts(settings: Settings, text: str, output: Path) -> Path:
    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}",
        headers={"xi-api-key": settings.elevenlabs_key, "Content-Type": "application/json",
                 "Accept": "audio/mpeg"},
        json={"text": text, "model_id": "eleven_multilingual_v2",
              "voice_settings": {"stability": 0.45, "similarity_boost": 0.8}},
        timeout=REQUEST_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(f"ElevenLabs {response.status_code}: {response.text[:400]}")
    output.write_bytes(response.content)
    return output


def render_segment(media: Path, kind: str, output: Path, duration: float,
                   width: int, height: int) -> None:
    ffmpeg = ffmpeg_path()
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,"
          f"crop={width}:{height},setsar=1,format=yuv420p")
    args = [ffmpeg, "-y"]
    args += (["-loop", "1", "-i", str(media)] if kind == "image"
             else ["-stream_loop", "-1", "-i", str(media)])
    args += ["-t", f"{duration}", "-vf", vf, "-r", "30", "-an", "-c:v", "libx264",
             "-preset", "veryfast", "-pix_fmt", "yuv420p", str(output)]
    run_command(args)


def concat_segments(segments: list[Path], output: Path) -> Path:
    ffmpeg = ffmpeg_path()
    list_file = output.parent / "concat.txt"
    list_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in segments), encoding="utf-8")
    run_command([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                 "-c", "copy", str(output)])
    return output


def mux_audio(video: Path, audio: Path, output: Path) -> Path:
    ffmpeg = ffmpeg_path()
    run_command([ffmpeg, "-y", "-i", str(video), "-i", str(audio),
                 "-filter_complex", "[1:a]afade=t=in:st=0:d=2[a]",
                 "-map", "0:v:0", "-map", "[a]", "-c:v", "libx264", "-c:a", "aac",
                 "-shortest", "-movflags", "+faststart", str(output)])
    return output


def render_project(settings: Settings, script: dict[str, Any], fmt: str,
                   tracker: StageTracker) -> Path:
    width, height = (1080, 1920) if fmt == "shorts" else (1920, 1080)
    portrait = fmt == "shorts"
    workdir = Path(tempfile.mkdtemp(prefix="ytstudio_"))
    try:
        shots = script["shots"]
        rendered: list[Path] = []
        tracker.start("media", 8, f"0/{len(shots)}")
        for index, shot in enumerate(shots):
            media, kind = get_media(settings, shot["visual_query"], workdir, index, portrait)
            tracker.update("media", 8 + int((index + 1) / len(shots) * 32),
                           f"{index + 1}/{len(shots)} · {kind}")
            duration = max(0.7, float(shot["end"]) - float(shot["start"]))
            segment = workdir / f"segment_{index}.mp4"
            render_segment(media, kind, segment, duration, width, height)
            rendered.append(segment)
        tracker.done("media", 44, f"{len(shots)} clips")
        tracker.done("segments", 52, f"{len(rendered)} segments")
        tracker.start("voice", 58, "ElevenLabs")
        audio = elevenlabs_tts(settings, script.get("narration", ""), workdir / "voiceover.mp3")
        tracker.done("voice", 70, f"{audio.stat().st_size // 1024} KB")
        tracker.start("merge", 76, "concat")
        silent = concat_segments(rendered, workdir / "silent.mp4")
        tracker.done("merge", 86)
        tracker.start("mux", 90, "afade + shortest")
        final = workdir / f"{safe_name(script.get('title', 'video'))}.mp4"
        mux_audio(silent, audio, final)
        outputs = Path("outputs")
        outputs.mkdir(exist_ok=True)
        destination = outputs / final.name
        shutil.copy2(final, destination)
        tracker.done("mux", 96, f"{destination.stat().st_size // 1024} KB")
        tracker.done("finish", 100)
        return destination
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


ASSISTANT_INSTRUCTIONS = """
You are the in-app assistant of YouTube Video Studio Pro.
Answer in {lang_name}. Be concrete and practical.
You help with: topics, hooks, scripts, monetisation risks, JSON script auditing, and errors from
Streamlit, Gemini/Claude, ElevenLabs, Pexels/Pixabay, YouTube Data API and FFmpeg.
You also correct the user's own wording and mistakes when asked.
If you fix the script, put one complete JSON object between CORRECTED_JSON_START and CORRECTED_JSON_END
(keys: title, hook, narration, description, tags, shots[start,end,voiceover,visual_query,on_screen_text]).
Never repeat API keys.
""".strip()


def call_assistant(settings: Settings, message: str, script: dict[str, Any] | None,
                   template: dict[str, Any] | None) -> str:
    api_key = settings.gemini_api_key or (
        settings.llm_api_key if settings.llm_provider == "Gemini" else "")
    if not api_key:
        raise ValueError("Gemini Assistant API key is missing.")
    lang_name = {"uz": "Uzbek", "ru": "Russian", "en": "English"}[settings.ui_lang]
    context = json.dumps(script, ensure_ascii=False, indent=2) if script else "No script yet."
    channel = json.dumps(template, ensure_ascii=False)[:2500] if template else "No channel analysis."
    prompt = (f"{ASSISTANT_INSTRUCTIONS.format(lang_name=lang_name)}\n\nSCRIPT:\n{context}\n\n"
              f"CHANNEL TEMPLATE:\n{channel}\n\nUSER:\n{message}")
    return gemini_generate(api_key, settings.gemini_model, prompt, False, 0.35)


def extract_corrected_json(text: str) -> dict[str, Any] | None:
    if "CORRECTED_JSON_START" not in text or "CORRECTED_JSON_END" not in text:
        return None
    segment = text.split("CORRECTED_JSON_START", 1)[1].split("CORRECTED_JSON_END", 1)[0]
    try:
        return normalize_script(parse_json_object(segment))
    except (ValueError, json.JSONDecodeError):
        return None


def sidebar_settings() -> Settings:
    with st.sidebar:
        ui_label = st.selectbox("Language / Til / Язык", list(LANGUAGES.keys()), index=0)
        lang = LANGUAGES[ui_label]
        st.markdown(f"### {tr('settings', lang)}")
        st.caption(tr("settings_note", lang))
        content_label = st.selectbox(tr("content_language", lang),
                                     [CONTENT_LANGUAGES[code][lang] for code in ("uz", "ru", "en")])
        content_lang = next(code for code in ("uz", "ru", "en")
                            if CONTENT_LANGUAGES[code][lang] == content_label)
        with st.expander(tr("grp_script_ai", lang), expanded=True):
            provider = st.selectbox(tr("provider", lang), ["Gemini", "Claude"])
            default_model = (DEFAULT_GEMINI_MODEL if provider == "Gemini"
                             else "claude-3-5-sonnet-latest")
            llm_model = st.text_input(tr("model", lang), value=default_model, key="llm_model")
            llm_key = st.text_input(tr("api_key", lang), type="password", key="llm_key")
        with st.expander(tr("grp_assistant", lang), expanded=True):
            assistant_key = st.text_input(f"Gemini {tr('api_key', lang)}", type="password",
                                          key="assistant_key")
            assistant_model = st.text_input(f"Gemini {tr('model', lang)}",
                                            value=DEFAULT_GEMINI_MODEL, key="assistant_model")
        with st.expander(tr("grp_voice", lang)):
            eleven_key = st.text_input(f"ElevenLabs {tr('api_key', lang)}", type="password")
            voice_id = st.text_input(tr("voice_id", lang), value="21m00Tcm4TlvDq8ikWAM")
        with st.expander(tr("grp_media", lang)):
            media_provider = st.selectbox(tr("provider", lang), ["Pexels", "Pixabay"])
            media_key = st.text_input(f"{media_provider} {tr('api_key', lang)}", type="password")
        with st.expander(tr("grp_youtube", lang)):
            youtube_key = st.text_input(f"YouTube Data {tr('api_key', lang)}", type="password")
        active = st.session_state.get("active_gemini_model")
        if active:
            st.success(f"{tr('active_model', lang)}: {active}")
        st.caption(tr("ffmpeg_note", lang))
    return Settings(lang, content_lang, provider, llm_key, llm_model, assistant_key,
                    assistant_model, eleven_key, voice_id, media_provider, media_key, youtube_key)


def render_steps(lang: str, script_ready: bool, approved: bool, video_ready: bool) -> None:
    def cls(done: bool, active: bool) -> str:
        return "ys-step done" if done else ("ys-step active" if active else "ys-step")

    st.markdown(
        f"""<div class="ys-steps">
<div class="{cls(script_ready, not script_ready)}">01<b>{tr('step_script', lang)}</b></div>
<div class="{cls(approved, script_ready and not approved)}">02<b>{tr('step_approve', lang)}</b></div>
<div class="{cls(video_ready, approved and not video_ready)}">03<b>{tr('step_render', lang)}</b></div>
<div class="{cls(video_ready, False)}">04<b>{tr('step_download', lang)}</b></div>
</div>""",
        unsafe_allow_html=True,
    )


def show_script_overview(script: dict[str, Any], lang: str) -> None:
    shots = script.get("shots", [])
    total = max((float(s.get("end", 0)) for s in shots), default=0)
    col1, col2, col3 = st.columns(3)
    col1.metric(tr("shots", lang), len(shots))
    col2.metric(tr("duration", lang), f"{total:.0f}s")
    col3.metric(tr("tags", lang), len(script.get("tags", [])))
    st.markdown(
        f"<div class='ys-card'><h3>{html.escape(str(script.get('title', '')))}</h3>"
        f"<p class='ys-tip'><b>Hook:</b> {html.escape(str(script.get('hook', '')))}</p></div>",
        unsafe_allow_html=True,
    )
    for shot in shots:
        st.markdown(
            f"""<div class="ys-shot">
<div class="t">{float(shot['start']):.0f}s - {float(shot['end']):.0f}s · {html.escape(shot.get('on_screen_text', ''))}</div>
<div class="v">{html.escape(shot.get('voiceover', ''))}</div>
<div class="q">{html.escape(shot.get('visual_query', ''))}</div>
</div>""",
            unsafe_allow_html=True,
        )


def script_stages(lang: str) -> list[tuple[str, str]]:
    return [("analyze", tr("stage_analyze", lang)), ("script", tr("stage_script", lang))]


def render_stages(lang: str) -> list[tuple[str, str]]:
    return [("media", tr("stage_media", lang)), ("segments", tr("stage_segments", lang)),
            ("voice", tr("stage_voice", lang)), ("merge", tr("stage_merge", lang)),
            ("mux", tr("stage_mux", lang)), ("finish", tr("stage_done", lang))]


def studio_tab(settings: Settings) -> None:
    lang = settings.ui_lang
    st.markdown(f"<div class='ys-card'><h3>{tr('idea', lang)}</h3></div>", unsafe_allow_html=True)
    topic = st.text_area(tr("topic_ph", lang), height=88, placeholder=tr("topic_ph", lang),
                         label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1:
        fmt_label = st.radio(tr("format", lang), ["Shorts · 9:16", "Long · 16:9"])
    with col2:
        mode = st.radio(tr("mode", lang), ["Full Auto", "Semi-Auto"])
    fmt = "shorts" if fmt_label.startswith("Shorts") else "long"
    template = st.session_state.get("channel_template")
    use_template = False
    if template:
        use_template = st.toggle(f"{tr('use_channel_style', lang)} - {template.get('title', '')}",
                                 value=True)
    demo = st.toggle(tr("demo_mode", lang), value=False)

    if st.button(tr("generate", lang), type="primary", use_container_width=True):
        if not topic.strip():
            st.error(tr("need_topic", lang))
        else:
            st.markdown(f"<p class='ys-tip'>{tr('pipeline', lang)}</p>", unsafe_allow_html=True)
            tracker = StageTracker(script_stages(lang), lang)
            tracker.mount()
            try:
                if demo:
                    tracker.start("analyze", 10)
                    tracker.done("analyze", 30, "demo")
                    tracker.start("script", 60, "demo")
                    script = demo_script(topic, settings.content_lang)
                    tracker.done("script", 100, f"{len(script['shots'])} shots")
                else:
                    block = (youtube_api.template_prompt_block(template)
                             if (use_template and template) else "")
                    script = generate_script(settings, topic, fmt, block, tracker)
                st.session_state.update(
                    script=script,
                    script_json=json.dumps(script, ensure_ascii=False, indent=2),
                    approved=(mode == "Full Auto"), fmt=fmt)
                st.success(tr("script_ready", lang))
            except Exception as exc:
                tracker.fail("script", str(exc))
                st.error(friendly_error(str(exc), lang))

    script = st.session_state.get("script")
    if not script:
        st.info(tr("empty_hint", lang))
        return

    st.divider()
    show_script_overview(script, lang)

    with st.expander(tr("advanced_json", lang)):
        edited = st.text_area("JSON", value=st.session_state.get("script_json", ""), height=300,
                              label_visibility="collapsed")
        if st.button(tr("save_json", lang), use_container_width=True):
            try:
                st.session_state["script"] = normalize_script(parse_json_object(edited))
                st.session_state["script_json"] = json.dumps(st.session_state["script"],
                                                             ensure_ascii=False, indent=2)
                st.session_state["approved"] = True
                st.success(tr("approved_msg", lang))
                st.rerun()
            except Exception as exc:
                st.error(f"JSON: {exc}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(tr("approve", lang), use_container_width=True):
            st.session_state["approved"] = True
            st.rerun()
    with col2:
        render_clicked = st.button(tr("render", lang), type="primary", use_container_width=True,
                                   disabled=not st.session_state.get("approved", False))

    if render_clicked:
        missing = [name for name, value in (
            ("ElevenLabs API key", settings.elevenlabs_key),
            (f"{settings.media_provider} API key", settings.media_api_key),
        ) if not value]
        if missing:
            st.error(f"{tr('missing_keys', lang)} " + ", ".join(missing))
        else:
            st.markdown(f"<p class='ys-tip'>{tr('pipeline', lang)}</p>", unsafe_allow_html=True)
            tracker = StageTracker(render_stages(lang), lang)
            tracker.mount()
            try:
                output = render_project(settings, st.session_state["script"],
                                        st.session_state.get("fmt", "shorts"), tracker)
                st.session_state["output"] = str(output)
                st.success(tr("video_ready", lang))
            except Exception as exc:
                tracker.fail("mux", str(exc))
                st.error(friendly_error(str(exc), lang))

    output = st.session_state.get("output")
    if output and Path(output).exists():
        st.divider()
        st.markdown(f"<div class='ys-card'><h3>{tr('result', lang)}</h3></div>",
                    unsafe_allow_html=True)
        st.video(output)
        st.download_button(tr("download_mp4", lang), data=Path(output).read_bytes(),
                           file_name=Path(output).name, mime="video/mp4",
                           type="primary", use_container_width=True)


def channel_tab(settings: Settings) -> None:
    lang = settings.ui_lang
    st.markdown(f"<div class='ys-card'><h3>{tr('channel_title', lang)}</h3>"
                f"<p class='ys-tip'>{tr('channel_sub', lang)}</p></div>", unsafe_allow_html=True)
    url = st.text_input(tr("channel_url", lang), placeholder="https://www.youtube.com/@channel")
    if st.button(tr("analyze", lang), type="primary", use_container_width=True):
        if not settings.youtube_api_key:
            st.error(tr("need_yt_key", lang))
        elif not url.strip():
            st.error(tr("need_topic", lang))
        else:
            try:
                with st.spinner(tr("analyzing", lang)):
                    channel = youtube_api.resolve_channel(settings.youtube_api_key, url)
                    videos = youtube_api.recent_videos(settings.youtube_api_key, channel)
                    template = youtube_api.build_template(channel, videos)
                st.session_state["channel_template"] = template
                st.success(tr("channel_saved", lang))
            except Exception as exc:
                st.error(friendly_error(str(exc), lang))

    template = st.session_state.get("channel_template")
    if not template:
        return
    col1, col2, col3 = st.columns(3)
    col1.metric(tr("subs", lang), f"{template['subscribers']:,}")
    col2.metric(tr("views", lang), f"{template['total_views']:,}")
    col3.metric(tr("videos", lang), f"{template['video_count']:,}")
    col4, col5, col6 = st.columns(3)
    col4.metric(tr("duration", lang), f"{template['avg_duration_sec']}s")
    col5.metric("Shorts %", f"{int(template['shorts_ratio'] * 100)}%")
    col6.metric(f"{tr('views', lang)} ~", f"{template['avg_views_recent']:,}")
    st.markdown(f"<div class='ys-card'><h3>{tr('top_videos', lang)}</h3></div>",
                unsafe_allow_html=True)
    for video in template.get("videos", [])[:8]:
        st.markdown(
            f"""<div class="ys-shot"><div class="t">{video['views']:,} views · {video['published']}</div>
<div class="v">{html.escape(video['title'])}</div>
<div class="q">{video['likes']:,} likes · {video['comments']:,} comments · {video['duration']}</div></div>""",
            unsafe_allow_html=True,
        )
    if template.get("top_tags"):
        st.markdown(f"**{tr('tags', lang)}:** " + ", ".join(template["top_tags"]))
    with st.expander(tr("channel_template", lang)):
        st.code(youtube_api.template_prompt_block(template), language="text")


def assistant_tab(settings: Settings) -> None:
    lang = settings.ui_lang
    st.markdown(f"<div class='ys-card'><h3>{tr('assistant_title', lang)}</h3>"
                f"<p class='ys-tip'>{tr('assistant_sub', lang)}</p></div>", unsafe_allow_html=True)
    st.session_state.setdefault("assistant_messages", [])
    col1, col2 = st.columns(2)
    with col1:
        audit = st.button(tr("audit", lang), use_container_width=True)
    with col2:
        if st.button(tr("clear_chat", lang), use_container_width=True):
            st.session_state["assistant_messages"] = []
            st.rerun()
    for message in st.session_state["assistant_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input(tr("assistant_ph", lang))
    if audit:
        question = ("Audit the current script: timestamp continuity, narration vs shots, hook "
                    "strength, visual_query quality, monetisation risks. Propose fixes and, if "
                    "needed, return the full corrected JSON between the CORRECTED_JSON markers.")
    if question:
        st.session_state["assistant_messages"].append({"role": "user", "content": question})
        try:
            with st.spinner(tr("assistant_working", lang)):
                answer = call_assistant(settings, question, st.session_state.get("script"),
                                        st.session_state.get("channel_template"))
            st.session_state["assistant_messages"].append({"role": "assistant", "content": answer})
            corrected = extract_corrected_json(answer)
            if corrected:
                st.session_state["assistant_corrected_json"] = corrected
        except Exception as exc:
            st.session_state["assistant_messages"].append(
                {"role": "assistant", "content": friendly_error(str(exc), lang)})
        st.rerun()
    corrected = st.session_state.get("assistant_corrected_json")
    if corrected:
        st.warning(tr("corrected_offer", lang))
        if st.button(tr("apply_corrected", lang), type="primary", use_container_width=True):
            st.session_state["script"] = corrected
            st.session_state["script_json"] = json.dumps(corrected, ensure_ascii=False, indent=2)
            st.session_state["approved"] = True
            st.session_state.pop("assistant_corrected_json", None)
            st.success(tr("applied", lang))
            st.rerun()


def integrations_tab(settings: Settings) -> None:
    lang = settings.ui_lang
    st.markdown(f"<div class='ys-card'><h3>{tr('int_title', lang)}</h3></div>",
                unsafe_allow_html=True)
    rows = [
        ("Gemini", bool(settings.llm_api_key or settings.gemini_api_key),
         "generativelanguage.googleapis.com"),
        ("Claude", bool(settings.llm_api_key and settings.llm_provider == "Claude"),
         "api.anthropic.com"),
        ("ElevenLabs", bool(settings.elevenlabs_key), "api.elevenlabs.io"),
        (settings.media_provider, bool(settings.media_api_key),
         "api.pexels.com" if settings.media_provider == "Pexels" else "pixabay.com/api"),
        ("YouTube Data API v3", bool(settings.youtube_api_key), "googleapis.com/youtube/v3"),
        ("FFmpeg", shutil.which("ffmpeg") is not None, "local binary"),
    ]
    for name, connected, endpoint in rows:
        pill = "on" if connected else "off"
        label = tr("int_connected", lang) if connected else tr("int_missing", lang)
        st.markdown(
            f"<div class='ys-row'><span class='n'>{html.escape(name)}"
            f"<span class='ys-pill {pill}'>{label}</span></span>"
            f"<span class='s'>{html.escape(endpoint)}</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown({
        "uz": "#### Ochiq API'si yo'q xizmatlar\n"
              "- **vidIQ** va **Nexlev** ommaviy API taklif qilmaydi, shuning uchun ularni "
              "to'g'ridan-to'g'ri ulash mumkin emas. Ma'lumotini faqat qo'lda eksport qilib, "
              "matn sifatida kiritish mumkin.\n"
              "- Raqobatchi kanal tahlili uchun rasmiy yo'l - **YouTube Data API v3**, u shu ilovada ishlaydi.\n"
              "- **MCP** serverlarini ulash uchun doimiy ishlaydigan backend kerak; bepul Streamlit "
              "hostingida bu barqaror ishlamaydi.",
        "ru": "#### Сервисы без публичного API\n"
              "- **vidIQ** и **Nexlev** не предоставляют публичный API, поэтому прямое подключение "
              "невозможно. Их данные можно экспортировать вручную и вставить текстом.\n"
              "- Официальный путь для анализа конкурентов - **YouTube Data API v3**, он уже работает здесь.\n"
              "- Для подключения **MCP**-серверов нужен постоянный backend; на бесплатном Streamlit "
              "это ненадёжно.",
        "en": "#### Services without a public API\n"
              "- **vidIQ** and **Nexlev** do not offer a public API, so they cannot be connected "
              "directly. Their data can only be exported manually and pasted as text.\n"
              "- The official route for competitor research is the **YouTube Data API v3**, already wired here.\n"
              "- Connecting **MCP** servers needs an always-on backend; free Streamlit hosting is "
              "not reliable for it.",
    }[lang])


def guide_tab(settings: Settings) -> None:
    lang = settings.ui_lang
    st.markdown(f"<div class='ys-card'><h3>{tr('guide_title', lang)}</h3></div>",
                unsafe_allow_html=True)
    st.markdown({
        "uz": "1. Sozlamalarda til va API kalitlarni kiriting.\n"
              "2. Kerak bo'lsa **Kanal tahlili** bo'limida raqobatchi kanalni tahlil qiling.\n"
              "3. **Studio**da mavzu yozing va ssenariy yarating.\n"
              "4. Ssenariyni tasdiqlang, so'ng MP4 render qiling.\n\n"
              "**Cheklovlar:** bepul hostingda render sekin; stock media litsenziyasini tekshiring; "
              "`503` xatosi vaqtinchalik server yuklamasi; API kalitlarini GitHub'ga joylamang.",
        "ru": "1. Укажите язык и API-ключи в настройках.\n"
              "2. При необходимости изучите канал конкурента во вкладке **Анализ канала**.\n"
              "3. В **Студии** введите тему и создайте сценарий.\n"
              "4. Подтвердите сценарий и запустите рендер MP4.\n\n"
              "**Ограничения:** на бесплатном хостинге рендер медленный; проверяйте лицензии "
              "сток-медиа; ошибка `503` - временная нагрузка; не публикуйте ключи в GitHub.",
        "en": "1. Set the language and API keys in settings.\n"
              "2. Optionally analyse a competitor in the **Channel analysis** tab.\n"
              "3. In **Studio**, enter a topic and generate the script.\n"
              "4. Approve the script, then render the MP4.\n\n"
              "**Limits:** rendering is slow on free hosting; verify stock media licences; "
              "`503` means temporary load; never commit API keys to GitHub.",
    }[lang])


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="V", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    settings = sidebar_settings()
    lang = settings.ui_lang
    st.markdown(
        f"""<div class="ys-hero">
<div class="ys-badge">{tr('hero_badge', lang)}</div>
<h1>{APP_TITLE}</h1>
<p>{tr('hero_sub', lang)}</p>
</div>""",
        unsafe_allow_html=True,
    )
    render_steps(lang, "script" in st.session_state, bool(st.session_state.get("approved")),
                 bool(st.session_state.get("output")))
    tabs = st.tabs([tr("tab_studio", lang), tr("tab_channel", lang), tr("tab_assistant", lang),
                    tr("tab_integrations", lang), tr("tab_guide", lang)])
    with tabs[0]:
        studio_tab(settings)
    with tabs[1]:
        channel_tab(settings)
    with tabs[2]:
        assistant_tab(settings)
    with tabs[3]:
        integrations_tab(settings)
    with tabs[4]:
        guide_tab(settings)


if __name__ == "__main__":
    main()
