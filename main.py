"""
Telegram Trading Analysis Bot using Gemini 2.5 Flash API.
Architecture: Async/Non-blocking with Thread Execution & Advanced Diagnostic Debugging.
Author: Senior Software Engineer & Code Architect
"""

import os
import io
import logging
import asyncio
import traceback
from typing import Optional
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
GEMINI_TIMEOUT = 45.0
MAX_TELEGRAM_MESSAGE_LENGTH = 4000

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

# Default Fallback Prompt
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
    """Synchronous execution inside worker thread to prevent Event Loop locks."""
    if not GEMINI_API_KEY:
        raise ValueError("کلید GEMINI_API_KEY در تنظیمات (Environment Variables) یافت نشد.")
    
    local_client = genai.Client(api_key=GEMINI_API_KEY)
    image_part = types.Part.from_bytes(
        data=jpeg_bytes,
        mime_type="image/jpeg"
    )
    
    response = local_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[image_part, prompt]
    )
    
    return response.text if response else ""


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
            "👑 **سلام! به ربات سیگنال طلای امین خوش اومدید.**\n\n"
            "📈 عکس چارت طلای مد نظرتون رو بفرستید تا براتون تحلیل کنم.",
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


# --- PHOTO HANDLER WITH DETAILED DIAGNOSTICS ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    status_message = await update.message.reply_text("📥 **تصویر دریافت شد؛ در حال پردازش اولیه...**", parse_mode="Markdown")

    try:
        photo_file = await update.message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        await status_message.edit_text("🧠 **در حال کالبدشکافی چارت و شناسایی زون‌های SMC...**", parse_mode="Markdown")

        # 1. پردازش تصویر در Thread
        loop = asyncio.get_running_loop()
        jpeg_bytes = await loop.run_in_executor(
            None, process_image_to_bytes, photo_bytes
        )

        # 2. ارسال به Gemini با مدیریت صحیح خطاها
        max_retries = 3
        response_text = ""
        captured_error = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"شروع درخواست به Gemini (تلاش {attempt})...")
                response_text = await asyncio.wait_for(
                    asyncio.to_thread(execute_gemini_request, jpeg_bytes, ADVANCED_PA_PROMPT),
                    timeout=GEMINI_TIMEOUT
                )
                if response_text:
                    logger.info("پاسخ از Gemini با موفقیت دریافت شد.")
                    break

            except Exception as err:
                captured_error = err
                logger.warning(f"تلاش {attempt} از {max_retries} با خطا مواجه شد: {err}")
                if attempt < max_retries:
                    await asyncio.sleep(2)

        if not response_text and captured_error:
            raise captured_error

        # 3. ارسال پاسخ نهایی
        if response_text:
            await safe_reply_text(status_message, response_text)
        else:
            await status_message.edit_text("⚠️ **پاسخی دریافت نشد. لطفاً مجدداً تصویر چارت را ارسال کنید.**", parse_mode="Markdown")

    except asyncio.TimeoutError:
        logger.error("Gemini API Request timed out.")
        await status_message.edit_text("⏱️ **خطای زمان‌بندی (Timeout):** پاسخ از هوش مصنوعی بیش از ۴۵ ثانیه طول کشید.")

    except APIError as api_err:
        logger.error(f"Gemini API Error: {api_err}")
        err_details = f"⚠️ **خطای Gemini API:**\n```\nType: {type(api_err).__name__}\nMessage: {str(api_err)}\n```"
        try:
            await status_message.edit_text(err_details, parse_mode="Markdown")
        except Exception:
            await status_message.edit_text(f"⚠️ **خطای Gemini API:**\n{str(api_err)}")

    except Exception as sys_err:
        tb_str = traceback.format_exc()
        logger.error(f"Full Error Traceback:\n{tb_str}")
        
        detailed_sys_msg = f"⚠️ **خطای دقیق سیستمی:**\n```python\n{tb_str[-2500:]}\n```"
        try:
            await status_message.edit_text(detailed_sys_msg, parse_mode="Markdown")
        except Exception:
            await status_message.edit_text(f"⚠️ **خطای سیستم:**\n{tb_str[-2000:]}")


# --- ENTRY POINT ---
def main() -> None:
    Thread(target=start_health_check_server, daemon=True).start()

    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable is missing. Bot shutting down.")
        return

    app: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot successfully initialized and running polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
