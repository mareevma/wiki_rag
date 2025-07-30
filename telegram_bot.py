import os
import re
import io
import asyncio
import logging
from pathlib import Path
from typing import Optional

from PIL import Image
from telegram import Update
from telegram.constants import ParseMode, ChatAction
from telegram.error import BadRequest, TimedOut
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)
from telegram.request import HTTPXRequest
from agentic_rag import chat

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_CHUNK = 4000
IMG_REGEX = re.compile(r'!\[(.*?)\]\((.*?)\)')

def escape_markdown_v2(text: str) -> str:
    escape_chars = r'\_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

def prepare_image(path: str) -> Optional[io.BytesIO]:
    try:
        if not os.path.exists(path):
            logger.error(f"Файл изображения не найден: {path}")
            return None
        img = Image.open(path)
        img.thumbnail((1280, 1280))
        bio = io.BytesIO()
        img.convert("RGB").save(bio, format="JPEG", quality=85)
        bio.seek(0)
        return bio
    except Exception as e:
        logger.error(f"Ошибка подготовки изображения {path}: {e}")
        return None

async def send_response_with_images(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    agent_response: str,
    tool_output: str
):
    if not update.effective_chat: return
    chat_id = update.effective_chat.id
    last_pos = 0
    source_files = re.findall(r"SourceFile='(.*?)'", tool_output)
    if not source_files and not IMG_REGEX.search(agent_response):
        await context.bot.send_message(
            chat_id, escape_markdown_v2(agent_response), parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    for match in IMG_REGEX.finditer(agent_response):
        start, end = match.span()
        text_part = agent_response[last_pos:start].strip()
        if text_part:
            await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
            await context.bot.send_message(
                chat_id, escape_markdown_v2(text_part), parse_mode=ParseMode.MARKDOWN_V2
            )
        relative_img_path = match.group(2)
        absolute_img_path = None
        for source_html_path_str in source_files:
            source_dir = Path(source_html_path_str).parent
            potential_path = (source_dir / relative_img_path).resolve()
            if potential_path.exists():
                absolute_img_path = str(potential_path)
                break
        if absolute_img_path:
            await context.bot.send_chat_action(chat_id, ChatAction.UPLOAD_PHOTO)
            photo_io = prepare_image(absolute_img_path)
            if photo_io:
                try:
                    # 🔥🔥🔥 ИЗМЕНЕНИЕ ЗДЕСЬ: убран параметр caption 🔥🔥🔥
                    await context.bot.send_photo(chat_id, photo=photo_io)
                except (BadRequest, TimedOut) as e:
                    logger.error(f"Ошибка отправки фото {absolute_img_path}: {e}")
            else:
                 await context.bot.send_message(chat_id, f"⚠️ Не удалось обработать: `{absolute_img_path}`")
        else:
            err_msg = f"⚠️ Не удалось найти файл: `{relative_img_path}`"
            await context.bot.send_message(chat_id, escape_markdown_v2(err_msg), parse_mode=ParseMode.MARKDOWN_V2)
        last_pos = end

    remaining_text = agent_response[last_pos:].strip()
    if remaining_text:
        await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
        await context.bot.send_message(
            chat_id, escape_markdown_v2(remaining_text), parse_mode=ParseMode.MARKDOWN_V2
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not (update.message and update.message.text and update.effective_chat): return
    chat_id = update.effective_chat.id
    user_input = update.message.text
    await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
    try:
        loop = asyncio.get_running_loop()
        response_dict = await loop.run_in_executor(None, chat, user_input)
        await send_response_with_images(
            update, context, response_dict["text"], response_dict["tool_output"]
        )
    except Exception as exc:
        logger.exception(f"Критическая ошибка: {exc}")
        await context.bot.send_message(chat_id, "Произошла внутренняя ошибка.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_chat: return
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Здравствуйте! Я — ассистент по технической документации. Задайте вопрос."
    )

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.critical("Переменная окружения TELEGRAM_BOT_TOKEN не задана.")
        return
    request = HTTPXRequest(connect_timeout=20, read_timeout=60)
    application = (
        ApplicationBuilder().token(token).request(request).build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Бот запущен.")
    application.run_polling()

if __name__ == "__main__":
    main()