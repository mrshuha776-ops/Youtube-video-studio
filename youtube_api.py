"""YouTube Data API v3 helpers: resolve a channel link and build a content template."""
from __future__ import annotations

import re
from typing import Any

import requests

API = "https://www.googleapis.com/youtube/v3"
TIMEOUT = 45


def _get(path: str, api_key: str, **params: Any) -> dict[str, Any]:
    params["key"] = api_key
    response = requests.get(f"{API}/{path}", params=params, timeout=TIMEOUT)
    if not response.ok:
        raise RuntimeError(f"YouTube API {response.status_code}: {response.text[:400]}")
    return response.json()


def parse_channel_ref(url: str) -> tuple[str, str]:
    """Return (kind, value) where kind is id | handle | query."""
    text = url.strip()
    if not text:
        raise ValueError("Kanal havolasi bo‘sh.")
    match = re.search(r"/channel/(UC[\w-]{20,})", text)
    if match:
        return "id", match.group(1)
    if re.fullmatch(r"UC[\w-]{20,}", text):
        return "id", text
    match = re.search(r"(?:youtube\.com/)?@([\w.\-]+)", text)
    if match:
        return "handle", match.group(1)
    match = re.search(r"youtube\.com/(?:c|user)/([\w.\-]+)", text)
    if match:
        return "handle", match.group(1)
    return "query", text


def resolve_channel(api_key: str, url: str) -> dict[str, Any]:
    kind, value = parse_channel_ref(url)
    parts = "snippet,statistics,contentDetails,brandingSettings"
    if kind == "id":
        data = _get("channels", api_key, part=parts, id=value)
    elif kind == "handle":
        data = _get("channels", api_key, part=parts, forHandle=value)
        if not data.get("items"):
            found = _get("search", api_key, part="snippet", q=value, type="channel", maxResults=1)
            items = found.get("items", [])
            if not items:
                raise RuntimeError("Kanal topilmadi.")
            channel_id = items[0]["snippet"]["channelId"]
            data = _get("channels", api_key, part=parts, id=channel_id)
    else:
        found = _get("search", api_key, part="snippet", q=value, type="channel", maxResults=1)
        items = found.get("items", [])
        if not items:
            raise RuntimeError("Kanal topilmadi.")
        data = _get("channels", api_key, part=parts, id=items[0]["snippet"]["channelId"])
    items = data.get("items", [])
    if not items:
        raise RuntimeError("Kanal topilmadi yoki yopiq.")
    return items[0]


def recent_videos(api_key: str, channel: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    uploads = channel.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
    if not uploads:
        return []
    playlist = _get("playlistItems", api_key, part="contentDetails", playlistId=uploads, maxResults=min(50, limit))
    ids = [i["contentDetails"]["videoId"] for i in playlist.get("items", []) if i.get("contentDetails", {}).get("videoId")]
    if not ids:
        return []
    details = _get("videos", api_key, part="snippet,statistics,contentDetails", id=",".join(ids[:limit]))
    videos = []
    for item in details.get("items", []):
        stats = item.get("statistics", {})
        snippet = item.get("snippet", {})
        videos.append(
            {
                "id": item.get("id", ""),
                "title": snippet.get("title", ""),
                "published": snippet.get("publishedAt", "")[:10],
                "tags": snippet.get("tags", [])[:12],
                "duration": item.get("contentDetails", {}).get("duration", ""),
                "views": int(stats.get("viewCount", 0) or 0),
                "likes": int(stats.get("likeCount", 0) or 0),
                "comments": int(stats.get("commentCount", 0) or 0),
            }
        )
    videos.sort(key=lambda v: v["views"], reverse=True)
    return videos


def iso_seconds(duration: str) -> int:
    match = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration or "")
    if not match:
        return 0
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def build_template(channel: dict[str, Any], videos: list[dict[str, Any]]) -> dict[str, Any]:
    snippet = channel.get("snippet", {})
    stats = channel.get("statistics", {})
    durations = [iso_seconds(v["duration"]) for v in videos if v.get("duration")]
    shorts = sum(1 for d in durations if 0 < d <= 60)
    tag_counts: dict[str, int] = {}
    for video in videos:
        for tag in video.get("tags", []):
            key = tag.strip().lower()
            if key:
                tag_counts[key] = tag_counts.get(key, 0) + 1
    top_tags = [tag for tag, _ in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:12]]
    return {
        "channel_id": channel.get("id", ""),
        "title": snippet.get("title", ""),
        "description": (snippet.get("description", "") or "")[:1200],
        "country": snippet.get("country", ""),
        "subscribers": int(stats.get("subscriberCount", 0) or 0),
        "total_views": int(stats.get("viewCount", 0) or 0),
        "video_count": int(stats.get("videoCount", 0) or 0),
        "avg_duration_sec": round(sum(durations) / len(durations)) if durations else 0,
        "shorts_ratio": round(shorts / len(durations), 2) if durations else 0.0,
        "top_tags": top_tags,
        "top_titles": [v["title"] for v in videos[:10]],
        "avg_views_recent": round(sum(v["views"] for v in videos) / len(videos)) if videos else 0,
        "videos": videos[:10],
    }


def template_prompt_block(template: dict[str, Any]) -> str:
    return (
        f"KANAL ANDOZASI\n"
        f"Nom: {template.get('title')}\n"
        f"Obunachilar: {template.get('subscribers')}\n"
        f"O‘rtacha davomiylik: {template.get('avg_duration_sec')} sek\n"
        f"Shorts nisbati: {template.get('shorts_ratio')}\n"
        f"O‘rtacha ko‘rish (oxirgi videolar): {template.get('avg_views_recent')}\n"
        f"Ommabop sarlavhalar: {'; '.join(template.get('top_titles', [])[:8])}\n"
        f"Tez-tez ishlatilgan taglar: {', '.join(template.get('top_tags', [])[:10])}\n"
        f"Kanal tavsifi: {(template.get('description') or '')[:500]}\n"
        "Shu kanalning sarlavha uslubi, hook usuli, davomiylik va mavzu yo‘nalishini andoza sifatida oling, "
        "lekin matnni ko‘chirmang va original kontent yarating."
    )
