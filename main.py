import os
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai
from PIL import Image
import io

# 1. تنظیمات سرور فرعی برای زنده نگه داشتن برنامه در Render
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running 24/7!")

def run_dummy_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), SimpleHTTPRequestHandler)
    server.serve_forever()

# اجرای سرور در پس‌زمینه
Thread(target=run_dummy_server, daemon=True).start()

# 2. دریافت کلیدها از تنظیمات Render
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    print("Error: Environment variables TELEGRAM_BOT_TOKEN or GEMINI_API_KEY are missing!")
    exit(1)

# تنظیم موتور هوش مصنوعی Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# 3. دستورات ربات تلگرام
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! عکس چارت طلا (XAUUSD) را بفرست تا تحلیل فنی کامل تحویل بگیری.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("چارت دریافت شد. در حال تحلیل با Gemini...")
    
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    image = Image.open(io.BytesIO(photo_bytes))
    
    prompt = "You are an expert XAUUSD trader. Analyze this chart image. Provide trend, key support/resistance levels, and potential buy/sell setup in Persian."
    
    try:
        response = model.generate_content([prompt, image])
        await update.message.reply_text(response.text)
    except Exception as e:
        await update.message.reply_text(f"خطا در تحلیل چارت: {e}")

# 4. اجرای اصلی ربات
def main():
    # پاک‌سازی فاصله و کتیشن‌های احتمالی دور توکن
    clean_token = TELEGRAM_BOT_TOKEN.strip().strip("'").strip('"')
    
    app = ApplicationBuilder().token(clean_token).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("Bot is starting...")
    app.run_polling()

if __name__ == '__main__':
    main()
