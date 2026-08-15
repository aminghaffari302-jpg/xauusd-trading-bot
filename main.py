import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai
from PIL import Image
import io

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

SYSTEM_PROMPT = """
تو یک دستیار ارشد معامله‌گری طلا (XAUUSD) بر اساس سبک Smart Money Concepts (SMC)، ICT و Price Action هستی.
وظیفه تو تحلیل تصاویر چارت ارسال شده و پاسخ به سوالات تریدینگ است.
همیشه زون‌های FVG، Order Block، Liquidity Sweeps و CHoCH را بررسی کن.
در صورت عدم تایید یا وجود اخبار پرریسک، معامله را ابطال (Invalidate) کن.
"""

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! ربات دستیار تحلیلی طلا (XAUUSD) آماده است. تصویر چارت یا سوال خود را بفرستید.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    response = model.generate_content([SYSTEM_PROMPT, user_text])
    await update.message.reply_text(response.text)

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("در حال پردازش تصویر چارت...")
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    
    image = Image.open(io.BytesIO(photo_bytes))
    response = model.generate_content([SYSTEM_PROMPT, "این چارت طلا را بر اساس SMC و ICT تحلیل کن:", image])
    
    await update.message.reply_text(response.text)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()
