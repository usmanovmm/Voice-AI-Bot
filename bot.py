import asyncio
import json
import os
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiohttp import web
from groq import Groq

# Переменные окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = os.getenv("ADMIN_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not BOT_TOKEN or not ADMIN_ID or not GROQ_API_KEY:
    raise ValueError("Заполните BOT_TOKEN, ADMIN_ID и GROQ_API_KEY в настройках!")

ADMIN_ID = int(ADMIN_ID)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
groq_client = Groq(api_key=GROQ_API_KEY)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 **Привет! Я голосовой ИИ-ассистент.**\n\n"
        "🎤 Отправьте мне **голосовое сообщение** на русском, узбекском или смешанном языке, "
        "и я автоматически распознаю его и сформирую готовую карточку заявки!"
    )


@dp.message(F.voice)
async def process_voice(message: Message):
    status_msg = await message.answer(
        "🎧 *Слушаю и распознаю голос...*", parse_mode="Markdown"
    )

    local_filename = f"voice_{message.voice.file_id}.ogg"

    try:
        # 1. Скачиваем голосовой файл
        file_info = await bot.get_file(message.voice.file_id)
        await bot.download_file(file_info.file_path, local_filename)

        # 2. Распознавание речи через Groq Whisper
        with open(local_filename, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(local_filename, audio_file.read()),
                model="whisper-large-v3",
                prompt="Разговор на узбекском и русском языках.",
            )

        recognized_text = transcription.text

        await status_msg.edit_text(
            "🧠 *Анализирую заявку с помощью Llama 3...*", parse_mode="Markdown"
        )

        # 3. Извлечение деталей через Llama 3
        prompt = f"""
Ты — менеджер отдела продаж. 
Ниже распознанный текст голосового сообщения от клиента (русский, узбекский или смешанный язык):

"{recognized_text}"

Верни ответ СТРОГО в формате JSON без каких-либо лишних символов со следующими ключами:
- "client_intent": суть заказа или вопроса клиента
- "details": ключевые детали (количество, даты, названия товаров, бюджет)
- "language_detected": язык речи (Русский, Узбекский, Смешанный)
- "urgency": срочность (Высокая, Средняя, Обычная)
"""

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        ai_response = completion.choices[0].message.content.strip()
        if ai_response.startswith("```"):
            ai_response = (
                ai_response.replace("```json", "").replace("```", "").strip()
            )

        data = json.loads(ai_response)

        # Удаляем временный файл
        if os.path.exists(local_filename):
            os.remove(local_filename)

        # Ответ клиенту
        await status_msg.edit_text(
            f"✅ **Голосовое сообщение распознано!**\n\n"
            f"📝 **Текст сообщения:**\n_{recognized_text}_",
            parse_mode="Markdown",
        )

        # Уведомление администратору
        admin_card = (
            f"🚨 **НОВАЯ ГОЛОСОВАЯ ЗАЯВКА**\n\n"
            f"👤 **От:** {message.from_user.full_name} (@{message.from_user.username or 'нет'})\n"
            f"🌐 **Язык:** {data.get('language_detected', 'Не определен')}\n"
            f"⚡ **Срочность:** {data.get('urgency', 'Обычная')}\n\n"
            f"📌 **Суть:** {data.get('client_intent')}\n"
            f"📦 **Детали:** {data.get('details')}\n\n"
            f"🎙 **Расшифровка:**\n_{recognized_text}_"
        )

        await bot.send_message(
            chat_id=ADMIN_ID, text=admin_card, parse_mode="Markdown"
        )

    except Exception as e:
        await status_msg.edit_text(f"❌ Ошибка обработки: {str(e)}")
        if os.path.exists(local_filename):
            os.remove(local_filename)


# Сервер для поддержания активности на Render
async def handle_ping(request):
    return web.Response(text="Voice Bot Active")


async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle_ping)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()


async def main():
    print("🚀 Третий бот (Voice AI) запущен!")
    await asyncio.gather(
        start_web_server(),
        dp.start_polling(bot),
    )


if __name__ == "__main__":
    asyncio.run(main())
