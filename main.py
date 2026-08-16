"""
Telegram Trading Analysis Bot using Gemini API (Complete Institutional Build).
"""

import os
import io
import logging
import asyncio
import traceback
from typing import List
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

# --- CONSTANTS & CONFIGURATION ---
MAX_IMAGE_DIMENSION = 1024
GEMINI_TIMEOUT = 45.0
MAX_TELEGRAM_MESSAGE_LENGTH = 4000

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

# مدل‌های پایدار و فعال گوگل
CANDIDATE_MODELS: List[str] = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

# پرامپت پیش‌فرض و کامل SMC / Price Action
DEFAULT_PROMPT = """You are a Lead Institutional Technical Analyst specializing in Price Action, Smart Money Concepts (SMC), and RTM for XAUUSD (Gold). 
Analyze this chart image with surgical precision and structural depth. Produce a clean, highly structured report entirely in Persian (Farsi).

CRITICAL DIRECTION FOR SETUPS:
- Scenario A+ MUST be an Intraday/Swing High-RR setup targeting major structural zones rather than micro-scalps. Avoid tiny TP ranges. Aim for substantial Risk-to-Reward (minimum 1:3 RR) with extended targets (TP1, TP2, TP3).

Use a professional, sharp, and visually clear tone with effective structural emojis (👑, 📊, 🎯, ⚖️, 🛑, 🚀, ⚠️, 🟩, 🟦, 🟧).

Follow this EXACT response structure in Persian:

👑 **اتاق تحلیل اختصاصی طلا | XAUUSD VIP Analysis**
---

📊 **۱. کالبدشکافی ساختار بازار (Market Structure & SMC)**
• **روند و مومنتوم:** [صعودی / نزولی / رنج] + بررسی توان خریداران و فروشندگان.
• **تغییر ماهیت و شکست ساختار (CHOCH / BMS):** مشخص کردن آخرین شکست‌های ساختاری.
• **نواحی نقدینگی و گپ‌ها (Liquidity & FVG):** شناسایی استخر نقدینگی (BSL/SSL) و گپ‌های ارزش منصفانه (Fair Value Gaps).
• **بلاک‌های سفارش فعال (Order Blocks):** مشخص کردن دقیق زون‌های تقاضا (Demand) یا عرضه (Supply).

---

⚖️ **۲. سطوح و زون‌های کلیدی (Key Zones & Levels)**
🔴 **زون‌های عرضه / مقاومت:** [اعداد دقیق visible روی چارت]
🟢 **زون‌های تقاضا / حمایت:** [اعداد دقیق visible روی چارت]
🌐 **محدوده تعادلی (Equilibrium / P&D):** وضعیت قیمت در زون‌های G&D (Premium vs Discount).

---

🎯 **۳. سناریوها و درجه‌بندی سیگنال‌های معاملاتی (Trading Setups)**

🟩 **سناریو A+ (سیگنال اصلی و گام بزرگ / هم‌جهت با روند ماژور):**
• **نوع پوزیشن:** [Long / Short]
• **محدوده ورود (EP):** [عدد دقیق یا زون مشخص]
• **حد ضرر (SL):** [عدد دقیق + دلیل ساختاری]
• **حد سود اول (TP1):** [تارگت اولیه / ریسک‌فری]
• **حد سود دوم (TP2):** [تارگت اصلی و ساختاری]
• **حد سود سوم (TP3):** [تارگت گسترده / گرفتن گام کامل روند]
• **نسبت ریسک به ریوارد (R:R):** [حداقل ۱ به ۳ به بالا]

🟦 **سناریو B (سیگنال میان‌مدت / پولبک و تأیید مجدد):**
• **نوع پوزیشن:** [Long / Short]
• **محدوده ورود (EP):** [عدد یا محدوده]
• **حد ضرر (SL):** [عدد دقیق]
• **حد سود (TP):** [اعداد دقیق]

🟧 **سناریو C (سیگنال پرریسک / اسکلپ سریع ثانیه‌ای):**
• **شرایط ورود:** [توضیح کوتاه]
• **محدوده ورود / SL / TP:** [مشخصات دقیق]

---

🛡️ **۴. مدیریت ریسک و توصیه استراتژیک (Trading Plan & Warning)**
• **توصیه حجم:** [مثال: حداکثر ۱ الی ۲ درصد ریسک کل حساب]
• **تأییدیه‌های لازم (Confirmations):** [نوع کندل بازگشتی یا شکست تایم‌پایین مورد نیاز]
• **هشدار مهم:** [توضیح مختصر در مورد اخبارهای پیش‌رو یا نوسانات طلا]
"""


def load_prompt(file_path: str = "prompt.txt") -> str:
    """بارگذاری پرامپت از فایل خارجی در صورت وجود"""
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                logger.info(f"Loaded prompt successfully from {file_path}")
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
    return DEFAULT_PROMPT.strip()


ADVANCED_PA_PROMPT = load_prompt()


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
def process_image(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=85)
        return output_buffer.getvalue()


def execute_gemini_request(jpeg_bytes: bytes, prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("متغیر GEMINI_API_KEY تنظیم نشده است.")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    image_part = types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")

    last_err = None
    for model_name in CANDIDATE_MODELS:
        try:
            logger.info(f"تلاش برای ارسال درخواست به مدل: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=[image_part, prompt]
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_err = e
            logger.warning(f"خطا در مدل {model_name}: {e}")
            continue

    if last_err:
        raise last_err
    raise RuntimeError("پاسخی از سرورهای گوگل دریافت نشد.")


def test_gemini_connection() -> str:
    if not GEMINI_API_KEY:
        raise ValueError("کلید GEMINI_API_KEY در تنظیمات یافت نشد.")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    last_err = None

    for model_name in CANDIDATE_MODELS:
        try:
            res = client.models.generate_content(model=model_name, contents="Say 'API Ready'")
            if res and res.text:
                return f"ارتباط با مدل {model_name} کاملاً برقرار است."
        except Exception as e:
            last_err = e
            continue

    if last_err:
        raise last_err
    raise RuntimeError("اتصال به هیچ‌یک از مدل‌ها برقرار نشد.")


async def safe_reply_text(status_message, text: str) -> None:
    chunks = [text[i:i + MAX_TELEGRAM_MESSAGE_LENGTH] for i in range(0, len(text), MAX_TELEGRAM_MESSAGE_LENGTH)]
    for index, chunk in enumerate(chunks):
        try:
            if index == 0:
                await status_message.edit_text(chunk, parse_mode="Markdown")
            else:
                await status_message.reply_text(chunk, parse_mode="Markdown")
        except Exception:
            if index == 0:
                await status_message.edit_text(chunk)
            else:
                await status_message.reply_text(chunk)


# --- ALL TELEGRAM COMMAND HANDLERS ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "👑 **ربات تحلیل تخصصی طلا (XAUUSD) فعال است.**\n\n"
            "▫️ برای تست سریع اتصال دستور /test را بزنید.\n"
            "▫️ برای دریافت راهنما دستور /help را ارسال کنید.\n"
            "▫️ جهت تحلیل، کافیست عکس چارت خود را ارسال کنید.",
            parse_mode="Markdown"
        )


async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    status_msg = await update.message.reply_text("🔍 **در حال بررسی اتصال به هوش مصنوعی گوگل...**", parse_mode="Markdown")
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, test_gemini_connection),
            timeout=15.0
        )
        await status_msg.edit_text(f"✅ **تست موفقیت‌آمیز بود!**\n\n`{result}`", parse_mode="Markdown")
    except Exception as e:
        await status_msg.edit_text(f"❌ **خطا در تست اتصال:**\n`{str(e)}`", parse_mode="Markdown")


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "📊 **دریافت تحلیل جدید**\n\n"
            "لطفاً تصویر چارت مورد نظر خود را ارسال کنید تا پردازش آغاز شود.",
            parse_mode="Markdown"
        )


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "🎯 **بخش سیگنال‌های معاملاتی**\n\n"
            "تصویر چارت را ارسال کنید؛ سناریوهای A+ و B همراه با نقاط دقیق EP، SL و TP صادر خواهند شد.",
            parse_mode="Markdown"
        )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "📊 **وضعیت سیستم:**\n\n"
            "• اتصال به سرور: 🟢 فعال\n"
            "• موتور هوش مصنوعی: Gemini 2.5/2.0 Flash\n"
            "• اتصال متاتریدر: به‌زودی در آپدیت بعدی...",
            parse_mode="Markdown"
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "❓ **راهنمای استفاده از ربات:**\n\n"
            "۱. یک عکس واضح از چارت طلا (XAUUSD) ارسال کنید.\n"
            "۲. چند ثانیه منتظر بمانید تا هوش مصنوعی چارت را آنالیز کند.\n"
            "۳. گزارش شامل کالبدشکافی ساختار SMC، زون‌های کلیدی و سناریوهای ورود/خروج را دریافت کنید.",
            parse_mode="Markdown"
        )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    status_message = await update.message.reply_text("🔹 [۱/۴] **دریافت تصویر از تلگرام...**", parse_mode="Markdown")

    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        
        await status_message.edit_text("🔹 [۲/۴] **بهینه‌سازی و فشرده‌سازی تصویر...**", parse_mode="Markdown")
        loop = asyncio.get_running_loop()
        jpeg_bytes = await loop.run_in_executor(None, process_image, photo_bytes)

        await status_message.edit_text("🔹 [۳/۴] **تحلیل ساختار بازار با Gemini AI...**", parse_mode="Markdown")
        
        response_text = await asyncio.wait_for(
            loop.run_in_executor(None, execute_gemini_request, jpeg_bytes, ADVANCED_PA_PROMPT),
            timeout=GEMINI_TIMEOUT
        )

        await status_message.edit_text("🔹 [۴/۴] **ارسال پاسخ به تلگرام...**", parse_mode="Markdown")
        
        if response_text and response_text.strip():
            await safe_reply_text(status_message, response_text)
        else:
            await status_message.edit_text("⚠️ **پاسخی از هوش مصنوعی دریافت نشد.**", parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error handling photo: {e}")
        await status_message.edit_text(f"❌ **خطا در پردازش چارت:**\n`{str(e)}`", parse_mode="Markdown")


def main() -> None:
    # شروع سرور سلامت برای رندر
    Thread(target=start_health_check_server, daemon=True).start()

    if not TELEGRAM_BOT_TOKEN:
        logger.critical("توکن TELEGRAM_BOT_TOKEN ست نشده است.")
        return

    app: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # ثبت تمامی دستورات تلگرام
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("test", test_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("help", help_command))
    
    # پردازش عکس‌ها
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot started successfully with all handlers.")
    app.run_polling()


if __name__ == "__main__":
    main()
