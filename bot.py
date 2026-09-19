import asyncio
import json
import os
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiohttp import web
from groq import Groq

# 1. Считывание переменных окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = os.getenv("ADMIN_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not BOT_TOKEN or not ADMIN_ID or not GROQ_API_KEY:
    raise ValueError(
        "ОШИБКА: Заполните BOT_TOKEN, ADMIN_ID и GROQ_API_KEY в настройках окружения!"
    )

ADMIN_ID = int(ADMIN_ID)

# Инициализация бота и клиента Groq
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
groq_client = Groq(api_key=GROQ_API_KEY)


# Команда /start
@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 **Привет! Я голосовой ИИ-ассистент.**\n\n"
        "🎤 Отправьте мне **голосовое сообщение** на русском, узбекском или смешанном языке. "
        "Я распознаю речь и автоматически сформирую готовую карточку заявки!"
    )


# Обработка голосовых сообщений
@dp.message(F.voice)
async def process_voice(message: Message):
    status_msg = await message.answer(
        "🎧 *Слушаю и распознаю голос...*", parse_mode="Markdown"
    )

    local_filename = f"voice_{message.voice.file_id}.ogg"

    try:
        # 1. Скачивание аудиофайла из Telegram
        file_info = await bot.get_file(message.voice.file_id)
        await bot.download_file(file_info.file_path, local_filename)

        # 2. Распознавание речи через Groq Whisper
        with open(local_filename, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=(local_filename, audio_file.read()),
                model="whisper-large-v3",
                prompt="Разговор на узбекском и русском языках.",
            )

        recognized_text = transcription.text.strip()

        if not recognized_text:
            await status_msg.edit_text(
                "⚠️ Не удалось распознать текст из голосового сообщения. Попробуйте записать еще раз."
            )
            if os.path.exists(local_filename):
                os.remove(local_filename)
            return

        await status_msg.edit_text(
            "🧠 *Анализирую заявку с помощью ИИ...*", parse_mode="Markdown"
        )

        # 3. Формирование промпта для структурирования
        prompt = f"""
Ты — профессиональный менеджер отдела продаж. 
Ниже приведен распознанный текст голосового сообщения от клиента (русский, узбекский или смешанный язык):

"{recognized_text}"

Верни ответ STRICTLY в формате JSON без какого-либо дополнительного текста со следующими полями:
- "client_intent": суть заказа или вопроса клиента
- "details": ключевые детали (количество, даты, названия товаров, бюджет, адрес и т.д.)
- "language_detected": язык сообщения (Русский, Узбекский, Смешанный)
- "urgency": срочность (Высокая, Средняя, Обычная)
"""

        # Список активных моделей Groq по приоритету
        candidate_models = [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
        ]

        completion = None
        used_model = None

        # Перебор моделей на случай отключения или лимита какой-либо из них
        for model_name in candidate_models:
            try:
                completion = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                used_model = model_name
                break
            except Exception as model_err:
                print(f"Модель {model_name} временно недоступна: {model_err}")
                continue

        if not completion:
            raise Exception("Ни одна из доступных нейросетей не ответила.")

        ai_response = completion.choices[0].message.content.strip()

                # Очистка JSON от возможных тегов markdown
        ai_response = ai_response.strip()
        if "```" in ai_response:
            ai_response = ai_response.replace("```json", "").replace("```", "").strip()

        data = json.loads(ai_response)
