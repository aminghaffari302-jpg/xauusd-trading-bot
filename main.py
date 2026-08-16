import os
import io
import logging
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from PIL import Image
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from google import genai

# تنظیمات Logging برای سرور Render
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# دریافت متغیرهای محیطی
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

MAX_IMAGE_DIMENSION = 1024
MAX_TELEGRAM_MESSAGE_LENGTH = 3900

# مقداردهی اولیه کلاینت گوگل
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# لیست مدل‌ها بر اساس اولویت
CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]

SMC_PROMPT = """
شما یک تحلیل‌گر ارشد سبک SMC (Smart Money Concepts) و Price Action برای طلا (XAUUSD) هستید.
لطفاً تصویر چارت معامله‌گری ارائه شده را با دقت کالبدشکافی کرده و گزارش کاملاً ساختاریافته به فارسی ارائه دهید:

👑 **تحلیل اختصاصی SMC | XAUUSD**
---
📊 **۱. ساختار بازار:** [روند، CHOCH/BMS]
⚖️ **۲. زون‌های کلیدی:** [Order Blocks و FVGهای فعال]
🎯 **۳. سناریوی معاملاتی:**
• محدوده ورود (EP)
• حد ضرر (SL)
• حد سودها (TP1, TP2, TP3)
• نسبت ریسک به ریوارد (R:R)
🛡️ **۴. مدیریت ریسک و توصیه استراتژیک**
"""

# --- HEALTH CHECK SERVER FOR RENDER ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format_str: str, *args) -> None:
        return

def start_health_check_server() -> None:
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    logger.info(f"Health check HTTP server running on port {port}")
    server.serve_forever()


# --- HELPER FUNCTIONS ---
def process_image(image_bytes: bytes) -> Image.Image:
    """بهینه‌سازی ابعاد عکس جهت افزایش سرعت پردازش"""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img.copy()

def _call_gemini_sync(model_name: str, prompt: str, image: Image.Image = None):
    """فراخوانی همگام در Thread مجزا"""
    contents = [image, prompt] if image else prompt
    response = client.models.generate_content(
        model=model_name,
        contents=contents
    )
    return response.text if response else ""

async def analyze_with_fallback(prompt: str, image: Image.Image = None):
    """تست غیرهمگام مدل‌ها تا دریافت اولین پاسخ موفق"""
    if not client:
        raise ValueError("کلید GEMINI_API_KEY تعریف نشده است.")

    last_error = None
    for model_name in CANDIDATE_MODELS:
        try:
            logger.info(f"تلاش برای تست مدل: {model_name}")
            text = await asyncio.to_thread(_call_gemini_sync, model_name, prompt, image)
            if text and text.strip():
                return text.strip(), model_name
        except Exception as e:
            logger.warning(f"مدل {model_name} پاسخ نداد: {e}")
            last_error = e
            continue
            
    raise Exception(f"هیچ‌کدام از مدل‌ها پاسخگو نبودند. آخرین خطا: {last_error}")

async def safe_reply_text(status_msg, text: str, prefix: str = "") -> None:
    """ارسال ایمن متن‌های طولانی جهت جلوگیری از خطای طول پیام تلگرام"""
    full_text = f"{prefix}\n\n{text}" if prefix else text
    chunks = [full_text[i:i + MAX_TELEGRAM_MESSAGE_LENGTH] for i in range(0, len(full_text), MAX_TELEGRAM_MESSAGE_LENGTH)]
    
    for index, chunk in enumerate(chunks):
        try:
            if index == 0:
                await status_msg.edit_text(chunk, parse_mode="Markdown")
            else:
                await status_msg.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            # در صورت بروز خطای Parse Mode، به متو ساده سوییچ می‌کند
            if index == 0:
                await status_msg.edit_text(chunk)
            else:
                await status_msg.reply_text(chunk)


# --- COMMAND HANDLERS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "سلام! ربات تحلیل چارت طلا (XAUUSD) فعال است.\n"
        "عکس چارت را ارسال کنید یا از دستور /test برای تست اتصال استفاده کنید."
    )

async def test_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("⏳ در حال ارزیابی مدل‌های Gemini...")
    try:
        text, used_model = await analyze_with_fallback("Test connection. Reply with 'OK'")
        await status_msg.edit_text(
            f"✅ **اتصال موفقیت‌آمیز بود!**\n\n"
            f"📌 **مدل فعال در پروژه شما:** `{used_model}`\n"
            f"💬 **پاسخ:** {text}",
            parse_mode="Markdown"
        )
    except Exception as e:
        await status_msg.edit_text(f"❌ **خطا در تست اتصال:**\n`{e}`", parse_mode="Markdown")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("⏳ در حال دریافت و تحلیل هوشمند تصویر...")
    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        
        # پردازش تصویر در Thread مجزا
        image = await asyncio.to_thread(process_image, photo_bytes)

        analysis, used_model = await analyze_with_fallback(SMC_PROMPT, image)
        
        prefix = f"📊 **تحلیل تکنیکال SMC** (مدل: `{used_model}`):"
        await safe_reply_text(status_msg, analysis, prefix=prefix)

    except Exception as e:
        logger.error(f"خطا در پردازش عکس: {e}")
        await status_msg.edit_text(f"❌ **خطا در تحلیل تصویر:**\n`{e}`", parse_mode="Markdown")


# --- MAIN ENTRY POINT ---
def main():
    # اجرای سرور Health Check در پس‌زمینه برای Render
    Thread(target=start_health_check_server, daemon=True).start()

    if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
        logger.error("متغیرهای TELEGRAM_BOT_TOKEN یا GEMINI_API_KEY ست نشده‌اند.")
        return

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("test", test_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("ربات آماده به کار است.")
    app.run_polling()

if __name__ == "__main__":
    main()
