import os
import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from google import genai
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image

# --- HTTP Server for Render Keeping Alive ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"OK")

def run_http_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# --- Config Gemini API (New Client) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# --- Bot Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام! ربات تحلیل چارت طلا (XAUUSD) آماده است.\n"
        "لطفاً تصویر چارت مورد نظر خود را ارسال کنید تا تحلیل دریافت کنید."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_message = await update.message.reply_text("در حال دریافت و تحلیل چارت توسط Gemini...")
    file_path = "chart.jpg"
    try:
        # Download photo
        photo_file = await update.message.photo[-1].get_file()
        await photo_file.download_to_drive(file_path)

        # Open image with Pillow
        img = Image.open(file_path)

        prompt = (
            "You are an expert technical analyst specializing in XAUUSD (Gold). "
            "Analyze this chart image in detail. Provide market trend, key support and resistance levels, "
            "and potential trading setups. Respond entirely in Persian (Farsi)."
        )

        # Retry logic for handling temporary 503 server overload
        max_retries = 3
        response = None
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-flash-latest',
                    contents=[img, prompt]
                )
                break
            except Exception as api_err:
                if "503" in str(api_err) and attempt < max_retries - 1:
                    await status_message.edit_text(f"سرور گوگل شلوغ است، تلاش مجدد ({attempt + 1}/{max_retries})...")
                    await asyncio.sleep(3)
                else:
                    raise api_err
        
        # Clean up local image file
        if os.path.exists(file_path):
            os.remove(file_path)

        if response and response.text:
            await status_message.edit_text(response.text)
        else:
            await status_message.edit_text("پاسخی از مدل دریافت نشد. دوباره تلاش کنید.")

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        await status_message.edit_text(f"خطا در تحلیل چارت: {str(e)}")

def main():
    # Start background HTTP server
    threading.Thread(target=run_http_server, daemon=True).start()

    # Telegram Bot Setup
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable is missing.")
        return

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Bot is starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
