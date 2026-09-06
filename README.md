---
title: YouTube Video Studio
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# YouTube Video Studio

Shaxsiy foydalanish uchun Streamlit + Python + FFmpeg asosidagi video generator.

## Imkoniyatlar

- Mavzu yoki YouTube kanal havolasidan ssenariy va shot planner yaratish.
- Claude yoki Gemini providerini qo‘lda tanlash.
- Pexels yoki Pixabay’dan stock video/rasm olish.
- ElevenLabs orqali multilingual voice-over yaratish.
- FFmpeg bilan 9:16 yoki 16:9 watermark-free MP4 render qilish.
- `afade=t=in:st=0:d=2` va `-shortest` bilan audio-video mux.
- Full Auto va Semi-Auto rejimlari.
- API kalitlari faylga yozilmaydi: ular faqat Streamlit sessiyasida turadi.

## Ishga tushirish

1. Python 3.10+ va FFmpeg o‘rnating. `ffmpeg -version` ishlashini tekshiring.
2. Virtual muhit yarating va bog‘liqliklarni o‘rnating:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\\Scripts\\activate
pip install -r requirements.txt
```

3. Ilovani ishga tushiring:

```bash
streamlit run app.py
```

4. Sidebar ichida AI, ElevenLabs va Pexels/Pixabay kalitlarini kiriting.
5. Mavzu kiriting, ssenariy yarating va render qiling.

## Muhim cheklovlar

- YouTube kanal tahlili bu versiyada kanal sahifasining ochiq HTML matniga asoslanadi; to‘liq analytics uchun YouTube Data API yoki alohida eksport kerak.
- Stock media litsenziyasini va monetizatsiya talablarini har bir media fayl uchun tekshiring.
- ElevenLabs, Claude/Gemini va Pexels/Pixabay’ning bepul tarif limitlari mavjud.
- Uchinchi tomon API’lari o‘z endpoint yoki model nomlarini o‘zgartirishi mumkin; model nomlarini Settings’dan almashtirish mumkin.
- Uzun ssenariylar uchun ElevenLabs matn uzunligi limitiga qarab narration’ni segmentlarga bo‘lish kerak bo‘lishi mumkin.
- Render vaqtinchalik fayllarni `/tmp` ichida, yakuniy MP4’ni esa `outputs/` ichida saqlaydi.

## Xavfsizlik

API kalitlarini GitHub yoki umumiy hostingga joylamang. Ilovani faqat o‘zingiz uchun ishlatsangiz ham, HTTPS va autentifikatsiyasiz internetga ochiq deploy qilmang.

## Gemini yordamchi

Interfeys pastida Gemini yordamchi mavjud. U:

- YouTube g‘oya, hook va ssenariy bo‘yicha savollarga javob beradi;
- hozirgi JSON ssenariyni tekshiradi;
- timestamp, narration, visual query va monetizatsiya xavflarini ko‘rsatadi;
- foydalanuvchi yozgan xatolarni tuzatadi;
- tuzatilgan JSON’ni bir tugma bilan asosiy ssenariyga qo‘llaydi.

Sidebar’dagi **Gemini Assistant API key** maydoniga Gemini kalitini kiriting. Agar asosiy AI provider sifatida Gemini tanlansa, asosiy Gemini kalitidan ham foydalanishi mumkin.
