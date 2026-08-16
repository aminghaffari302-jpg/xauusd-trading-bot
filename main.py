import os
import io
import logging
import asyncio
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from PIL import Image
from google import genai
from google.genai import types
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- CONSTANTS & CONFIG --
MAX_IMAGE_DIMENSION = 1024
GEMINI_TIMEOUT = 45.0
MAX_TELEGRAM_MESSAGE_LENGTH = 4000

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

# مدل‌های رسمی فعال
CANDIDATE_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

DEFAULT_PROMPT = """You are a Lead Institutional Technical Analyst specializing in Price Action, Smart Money Concepts (SMC), and RTM for XAUUSD (Gold). 
Analyze this chart image with surgical precision and structural depth. Produce a clean, highly structured report entirely in Persian (Farsi).

CRITICAL DIRECTION FOR SETUPS:
- Scenario A+ MUST be an Intraday/Swing High-RR setup targeting major structural zones rather than micro-scalps. Minimum 1:3 RR with extended targets (TP1, TP2, TP3).

Follow this EXACT response structure in Persian:

👑 **اتاق تحلیل اختصاصی طلا | XAUUSD VIP Analysis**
---

📊 **۱. کالبدشکافی ساختار بازار (Market Structure & SMC)**
• **روند و مومنتوم:** [صعودی / نزولی / رنج]
• **تغییر ماهیت و شکست ساختار (CHOCH / BMS):** [بررسی دقیق]
• **نواحی نقدینگی و گپ‌ها (Liquidity & FVG):** [شناسایی زون‌ها]
• **بلاک‌های سفارش فعال (Order Blocks):** [محدوده‌ها]

---

⚖️ **۲. سطوح و زون‌های کلیدی (Key Zones & Levels)**
🔴 **زون‌های عرضه / مقاومت:** [اعداد روی چارت]
🟢 **زون‌های تقاضا / حمایت:** [اعداد روی چارت]

---

🎯 **۳. سناریوها و درجه‌بندی سیگنال‌های معاملاتی (Trading Setups)**

🟩 **سناریو A+ (سیگنال اصلی / هم‌جهت با روند ماژور):**
• **نوع پوزیشن:** [Long / Short]
• **محدوده ورود (EP):** [عدد دقیق]
• **حد ضرر (SL):** [عدد دقیق]
• **حد سودها:** [TP1 / TP2 / TP3]
• **نسبت ریسک به ریوارد:** [حداقل ۱ به ۳]

🟦 **سناریو B (سیگنال میان‌مدت / پولبک):**
• **مشخصات کامل:** [ورود / SL / TP]

---

🛡️ **۴. مدیریت ریسک:**
• **حجم پیشنهادی:** [۱ الی ۲ درصد ریسک]
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
    server.serve_forever()

# --- HELPER FUNCTIONS ---
def process_image(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

def execute_gemini_request(jpeg_bytes: bytes, prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("متغیر GEMINI_API_KEY در Render تنظیم نشده است.")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

    last_err = None
    for model_name in CANDIDATE_MODELS:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=[image_part, prompt]
            )
            if res and res.text:
                return res.text
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise RuntimeError("پاسخی از مدل‌ها دریافت نشد.")

def test_gemini_connection() -> str:
    if not GEMINI_API_KEY:
        raise ValueError("کلید GEMINI_API_KEY یافت نشد.")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_err = None

    for model_name in CANDIDATE_MODELS:
        try:
            res = client.models.generate_content(model=model_name, contents="Ping")
            if res and res.text:
                return f"اتصال به مدل {model_name} برقرار است."
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise RuntimeError("هیچ مدلی پاسخ نداد.")

async def safe_send(status_msg, text: str) -> None:
    chunks = [text[i:i + MAX_TELEGRAM_MESSAGE_LENGTH] for i in range(0, len(text), MAX_TELEGRAM_MESSAGE_LENGTH)]
    for idx, chunk in enumerate(chunks):
        try:
            if idx == 0:
                await status_msg.edit_text(chunk, parse_mode="Markdown")
            else:
                await status_msg.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            if idx == 0:
                await status_msg.edit_text(chunk)
            else:
                await status_msg.reply_text(chunk)

# --- BOT HANDLERS ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text("👑 **ربات تحلیل طلا آماده است.**\n\nبرای تست اتصال /test را بزنید یا عکس چارت بفرستید.")

async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    msg = await update.message.reply_text("🔍 در حال تست ارتباط...")
    try:
        loop = asyncio.get_running_loop()
        res = await asyncio.wait_for(loop.run_in_executor(None, test_gemini_connection), timeout=15.0)
        await msg.edit_text(f"✅ **وضعیت:**\n`{res}`", parse_mode="Markdown")
    except Exception as e:
        await msg.edit_text(f"❌ **خطا در تست:**\n`{str(e)}`", parse_mode="Markdown")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    status = await update.message.reply_text("⏳ در حال دریافت و تحلیل چارت...")
    try:
        photo = await update.message.photo[-1].get_file()
        p_bytes = await photo.download_as_bytearray()
        
        loop = asyncio.get_running_loop()
        jpeg_bytes = await loop.run_in_executor(None, process_image, p_bytes)
        
        reply = await asyncio.wait_for(
            loop.run_in_executor(None, execute_gemini_request, jpeg_bytes, DEFAULT_PROMPT),
            timeout=GEMINI_TIMEOUT
        )
        
        await safe_send(status, reply)
    except Exception as e:
        await status.edit_text(f"❌ **خطا در پردازش:**\n`{str(e)}`", parse_mode="Markdown")

def main() -> None:
    Thread(target=start_health_check_server, daemon=True).start()
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("توکن تلگرام موجود نیست.")
        return

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("test", test_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
