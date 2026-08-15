"""
Telegram Trading Analysis Bot using Gemini API.
Features: Micro-Diagnostic Logging, Pre-flight Testing, Dynamic Model Fallback, and Full Error Tracing.
Author: Senior Software Engineer & Code Architect
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
from google.genai.errors import APIError
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
GEMINI_TIMEOUT = 45.0  # زمان پاسخ‌دهی ۴۵ ثانیه‌ای برای تحلیل‌های سنگین SMC
MAX_TELEGRAM_MESSAGE_LENGTH = 4000

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

# لیست مدل‌های رسمی و رسمی پایداری که در API فعال هستند
PRIMARY_MODEL = os.environ.get("GEMINI_MODEL", "").strip()
raw_candidates = [PRIMARY_MODEL, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
CANDIDATE_MODELS: List[str] = list(dict.fromkeys([m for m in raw_candidates if m]))

# Default Prompt Strategy
DEFAULT_PROMPT = """You are a Lead Institutional Technical Analyst specializing in Price Action, Smart Money Concepts (SMC), and RTM for XAUUSD (Gold). 
Analyze this chart image with surgical precision and structural depth. Produce a clean, highly structured report entirely in Persian (Farsi).

CRITICAL DIRECTION FOR SETUPS:
- Scenario A+ MUST be an Intraday/Swing High-RR setup targeting major structural zones rather than micro-scalps. Avoid tiny TP ranges (e.g., small 60-pip moves). Aim for substantial Risk-to-Reward (minimum 1:3 RR) with extended targets (TP1, TP2, TP3).

Use a professional, sharp, and visually clear tone with effective structural emojis (such as 👑, 📊, 🎯, ⚖️, 🛑, 🚀, ⚠️, 🟩, 🟦, 🟧).

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
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                logger.info(f"Loaded prompt successfully from {file_path}")
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}. Using default prompt.")
    else:
        logger.info(f"File {file_path} not found. Using embedded default prompt.")
    return DEFAULT_PROMPT.strip()


ADVANCED_PA_PROMPT = load_prompt()


# --- HEALTH CHECK SERVER ---
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
def process_image_to_bytes(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="JPEG", quality=85)
        return output_buffer.getvalue()


def execute_gemini_request(jpeg_bytes: bytes, prompt: str) -> str:
    """Execution with active model fallback list."""
    if not GEMINI_API_KEY:
        raise ValueError("متغیر GEMINI_API_KEY در Environment Variables مقداردهی نشده است.")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    image_part = types.Part.from_bytes(
        data=jpeg_bytes,
        mime_type="image/jpeg"
    )

    last_err = None
    for model_name in CANDIDATE_MODELS:
        try:
            logger.info(f"در حال ارسال درخواست به مدل: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=[image_part, prompt]
            )
            if response and response.text:
                logger.info(f"تحلیل با موفقیت از مدل {model_name} دریافت شد.")
                return response.text
        except Exception as e:
            last_err = e
            logger.warning(f"عدم موفقیت در فراخوانی مدل {model_name}: {e}. سوییچ به مدل بعدی...")
            continue

    if last_err:
        raise last_err
    return ""


def test_gemini_connection() -> str:
    if not GEMINI_API_KEY:
        raise ValueError("متغیر GEMINI_API_KEY در Environment Variables مقداردهی نشده است.")
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    for model_name in CANDIDATE_MODELS:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents="Say 'API Connection Successful' in English."
            )
            if res and res.text:
                return f"[{model_name}]: {res.text.strip()}"
        except Exception:
            continue
            
    return "هیچ مدلی پاسخ نداد. لطفاً API Key خود را بررسی کنید."


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


# --- COMMAND HANDLERS ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "👑 **ربات تحلیل طلا فعال است.**\n\n"
            "▫️ برای تست سریع اتصال هوش مصنوعی دستور /test را بزنید.\n"
            "▫️ برای تحلیل چارت، عکس چارت طلا را ارسال کنید.",
            parse_mode="Markdown"
        )


async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message:
        return
    status_msg = await update.message.reply_text("🔍 **در حال تست ارتباط با API جمینای...**", parse_mode="Markdown")
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, test_gemini_connection),
            timeout=15.0
        )
        await status_msg.edit_text(f"✅ **تست موفقیت‌آمیز بود!**\n\nپاسخ گوگل:\n`{result.strip()}`", parse_mode="Markdown")
    except Exception as e:
        tb = traceback.format_exc()
        await status_msg.edit_text(
            f"❌ **تست اتصال با خطا مواجه شد!**\n\n"
            f"**نوع خطا:** `{type(e).__name__}`\n"
            f"**جزئیات:**\n```python\n{tb[-1500:]}\n```",
            parse_mode="Markdown"
        )


async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "📊 **دریافت جدیدترین تحلیل طلا**\n\n"
            "لطفاً تصویر چارت مورد نظر خود را ارسال کنید تا تحلیل SMC اختصاصی آن تولید شود.",
            parse_mode="Markdown"
        )


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "🎯 **دریافت وضعیت سیگنال‌های فعلی**\n\n"
            "سیگنال‌های معاملاتی فعال بر روی چارت‌های ارسالی پردازش می‌شوند. عکس چارت جدید خود را بفرستید.",
            parse_mode="Markdown"
        )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "📊 **وضعیت اتصال به متاتریدر (MetaTrader):**\n\n"
            "بزودی در بروزرسانی‌های آینده اضافه میشه😊",
            parse_mode="Markdown"
        )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "❓ **راهنمای استفاده از ربات:**\n\n"
            "۱. یک عکس واضح از چارت طلا (XAUUSD) ارسال کنید.\n"
            "۲. هوش مصنوعی ساختار بازار، زون‌ها و سناریوهای A+، B و C را محاسبه می‌کند.\n"
            "۳. سیگنال‌های ورودی به همراه TP و SL را دریافت کنید.",
            parse_mode="Markdown"
        )


# --- PHOTO HANDLER WITH STEP-BY-STEP DIAGNOSTICS ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    status_message = await update.message.reply_text("🔹 [۱/۵] **دریافت فایل تصویر از تلگرام...**", parse_mode="Markdown")

    try:
        # گام ۱: دانلود تصویر
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        
        # گام ۲: فشرده‌سازی
        await status_message.edit_text("🔹 [۲/۵] **تغییر ابعاد و فشرده‌سازی عکس...**", parse_mode="Markdown")
        loop = asyncio.get_running_loop()
        jpeg_bytes = await loop.run_in_executor(None, process_image_to_bytes, photo_bytes)

        # گام ۳: بررسی API Key
        await status_message.edit_text("🔹 [۳/۵] **اعتبارسنجی تنظیمات کلید API...**", parse_mode="Markdown")
        if not GEMINI_API_KEY:
            raise ValueError("متغیر GEMINI_API_KEY روی سرور تعریف نشده است.")

        # گام ۴: ارسال درخواست به Gemini
        await status_message.edit_text("🔹 [۴/۵] **کالبدشکافی چارت با هوش مصنوعی (حداکثر ۴۵ ثانیه)...**", parse_mode="Markdown")
        
        response_text = await asyncio.wait_for(
            loop.run_in_executor(None, execute_gemini_request, jpeg_bytes, ADVANCED_PA_PROMPT),
            timeout=GEMINI_TIMEOUT
        )

        # گام ۵: نمایش پاسخ
        await status_message.edit_text("🔹 [۵/۵] **دریافت پاسخ؛ در حال تنظیم قالب تلگرام...**", parse_mode="Markdown")
        
        if response_text and response_text.strip():
            await safe_reply_text(status_message, response_text)
        else:
            await status_message.edit_text("⚠️ **پاسخ دریافتی از Gemini خالی بود.**", parse_mode="Markdown")

    except asyncio.TimeoutError:
        logger.error("Gemini Timeout Error")
        await status_message.edit_text(
            "🛑 **خطای توقف زمان (Timeout Error):**\n"
            "ارسال درخواست به گوگل بیش از ۴۵ ثانیه طول کشید. احتمالاً ترافیک گوگل بالاست یا سرور با کندی شبکه مواجه شده است.",
            parse_mode="Markdown"
        )

    except APIError as api_err:
        logger.error(f"Gemini API Error: {api_err}")
        err_msg = (
            f"❌ **خطای رسمی Gemini API:**\n"
            f"▫️ **نوع خطا:** `{type(api_err).__name__}`\n"
            f"▫️ **متن پیام:**\n```\n{str(api_err)}\n```"
        )
        await status_message.edit_text(err_msg, parse_mode="Markdown")

    except Exception as sys_err:
        tb_str = traceback.format_exc()
        logger.error(f"Full Error Traceback:\n{tb_str}")
        err_msg = (
            f"❌ **خطای سیستمی رخ داد:**\n"
            f"▫️ **نوع استثنا:** `{type(sys_err).__name__}`\n"
            f"▫️ **ردیابی خط کد (Traceback):**\n```python\n{tb_str[-2000:]}\n```"
        )
        try:
            await status_message.edit_text(err_msg, parse_mode="Markdown")
        except Exception:
            await status_message.edit_text(f"❌ **خطا در اجرای برنامه:**\n{str(sys_err)}")


# --- ENTRY POINT ---
def main() -> None:
    Thread(target=start_health_check_server, daemon=True).start()

    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable is missing. Bot shutting down.")
        return

    app: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("test", test_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot started successfully with diagnostic tools and fallback models.")
    app.run_polling()


if __name__ == "__main__":
    main()
