import os
import io
import logging
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from PIL import Image
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from google import genai


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

PORT = int(os.environ.get("PORT", "8080"))

MAX_IMAGE_DIMENSION = 1536
MAX_TELEGRAM_MESSAGE_LENGTH = 3900


# =========================================================
# GEMINI CLIENT
# =========================================================

client = None

if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")


# =========================================================
# MODELS
# =========================================================

CANDIDATE_MODELS = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-2.5-pro",
    "gemini-3.1-flash-lite"
]


# =========================================================
# PROFESSIONAL SMC & NDS PROMPT
# =========================================================

SMC_PROMPT = """
تو یک الگوریتم معاملاتی فوق‌العاده حرفه‌ای، تریدر ارشد پرایس اکشن، اسمارت مانی (SMC) و NDS در بازار جهانی طلا (XAUUSD) هستی.
ماموریت تو کالبدشکافی موشکافانه، دقیق و حساب‌شده‌ی تصویر چارت ارسال‌شده است. پاسخ خود را با بالاترین کیفیت، لحن کاملاً حرفه‌ای و با استفاده از ایموجی‌های تخصصی در بخش‌های زیر ارائه بده:

👑 **گزارش جامع و مهندسی‌شده | تحلیل طلای امین**

---

🌐 **۱. بررسی وضعیت کلی بازار و ساختار (Market Structure)**
• **روند قالب (Trend):** [صعودی / نزولی / رنج - با ذکر دلیل ساختاری]
• **تایم‌فریم و مقیاس:** [تشخیص تایم‌فریم یا ساختار حرکتی از روی چارت]
• **وضعیت حرکتی قیمت:** [آیا قیمت در حال ایمپالس است یا اصلاح (Correction)؟]

---

💧 **۲. بررسی مناطق نقدینگی و سطوح کلیدی (Liquidity & Key Zones)**
• **استخرهای نقدینگی (Liquidity Pools):** [شناسایی دقیق نقدینگی‌های انباشته شده (BSL / SSL)]
• **مقاومت‌ها و حمایت‌های مهم:** [نقاط کلیدی واکنشی روی چارت]
• **بررسی نفوذ نقدینگی (Sweep):** [آیا نقدینگی سطحی جارو شده یا خیر؟]

---

📊 **۳. کالبدشکافی پرایس اکشن، اسمارت مانی و NDS**
• **تحلیل پرایس اکشن و کندل‌ها:** [مومنتوم کندل‌ها، وجود الگوهای برگشتی یا ادامه دهنده]
• **ردپای اسمارت مانی (SMC):** [محل دقیق BOS یا CHoCH های تایید شده]
• **نواحی عرضه و تقاضا (NDS & OB/FVG):** [بلاک سفارش‌های معتبر یا شکاف‌های نقدینگی فعال]

---

🎯 **۴. سناریوهای معاملاتی و سیگنال تخصصی**
• **سناریوی اصلی:** [تشریح کامل روند منطقی پیش رو]
• **سناریوی جایگزین (هايدن):** [شرایط تغییر جهت یا ابطال سناریوی اول]

⭐️ **سیگنال پیشنهادی ترید:**
• **جهت پوزیشن:** [🟢 BUY / 🔴 SELL / ⚪ NO TRADE]
• **رتبه کیفیت ستاپ (Grade):** [A+ / A / B+ / B / C+ / C] *(بر اساس قدرت ریسک و اعتبار زون)*
• **دلیل منطقی ورود (Rationale):** [توضیح فنی کوتاه که چرا این نقطه برای ورود انتخاب شده است]
• **محدوده ورود (Entry Zone - EP):** [عدد دقیق یا بازه قیمتی ورود]
• **حد ضرر (Stop Loss - SL):** [عدد دقیق محاسباتی پشت ناحیه]
• **حد سود اول (TP1):** [تارگت اول با ریسک به ریوارد استاندارد]
• **حد سود دوم (TP2):** [تارگت دوم پیشرفته]
• **حد سود نهایی (TP3):** [تارگت نهایی در لبه نقدینگی بعدی]
• **نسبت ریسک به ریوارد (R:R):** [مثلاً 1:2.5 یا 1:3]

---

🛡️ **۵. مدیریت ریسک و توصیه‌های حیاتی**
• **شرط ابطال تحلیل (Invalidation):** [چه سطحی اگر شکسته شود کل این تحلیل فاقد اعتبار است؟]
• **استراتژی مدیریت معامله:** [توصیه برای ریسک‌فری کردن یا ورود پله‌ای]
• **هشدار معاملاتی:** [چه زمانی نباید وارد بازار شد؟]
"""


# =========================================================
# HEALTH CHECK SERVER
# =========================================================

class HealthCheckHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format_str, *args):
        return


def start_health_check_server():
    try:
        server = HTTPServer(
            ("0.0.0.0", PORT),
            HealthCheckHandler
        )
        logger.info(f"Health check server running on port {PORT}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Health check server failed: {e}")


# =========================================================
# IMAGE PROCESSING
# =========================================================

def process_image(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        img.thumbnail(
            (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
            Image.Resampling.LANCZOS
        )

        if img.mode != "RGB":
            img = img.convert("RGB")

        return img.copy()


# =========================================================
# GEMINI SYNC CALL
# =========================================================

def _call_gemini_sync(
    model_name: str,
    prompt: str,
    image: Image.Image = None
):
    if not client:
        raise RuntimeError("Gemini client is not initialized.")

    contents = [image, prompt] if image is not None else prompt

    response = client.models.generate_content(
        model=model_name,
        contents=contents
    )

    if not response:
        return ""

    text = getattr(response, "text", None)
    return text.strip() if text else ""


# =========================================================
# GET AVAILABLE MODELS
# =========================================================

def _get_available_models_sync():
    if not client:
        return []

    available = []
    try:
        for model in client.models.list():
            name = getattr(model, "name", "")
            if name.startswith("models/"):
                name = name.replace("models/", "", 1)
            if "gemini" not in name.lower():
                continue
            available.append(name)
    except Exception as e:
        logger.error(f"Could not list Gemini models: {e}")

    return available


# =========================================================
# GEMINI ANALYSIS WITH FALLBACK
# =========================================================

async def analyze_with_fallback(
    prompt: str,
    image: Image.Image = None
):
    if not client:
        raise RuntimeError(
            "GEMINI_API_KEY تعریف نشده یا Gemini Client ساخته نشده است."
        )

    last_error = None

    for model_name in CANDIDATE_MODELS:
        try:
            logger.info(f"Trying Gemini model: {model_name}")

            text = await asyncio.to_thread(
                _call_gemini_sync,
                model_name,
                prompt,
                image
            )

            if text:
                logger.info(f"Gemini success with model: {model_name}")
                return text, model_name

            logger.warning(f"{model_name} returned empty response.")

        except Exception as e:
            last_error = e
            logger.warning(f"Model {model_name} failed: {e}")

    raise RuntimeError(
        "هیچ‌کدام از مدل‌های Gemini پاسخگو نبودند.\n"
        f"آخرین خطا: {last_error}"
    )


# =========================================================
# TELEGRAM MESSAGE SPLITTER
# =========================================================

def split_text(text: str, max_length: int = 3900):
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""
    paragraphs = text.split("\n")

    for paragraph in paragraphs:
        if len(paragraph) > max_length:
            if current:
                chunks.append(current)
                current = ""

            for i in range(0, len(paragraph), max_length):
                chunks.append(paragraph[i:i + max_length])
            continue

        proposed = (
            current + "\n" + paragraph
            if current
            else paragraph
        )

        if len(proposed) > max_length:
            if current:
                chunks.append(current)
            current = paragraph
        else:
            current = proposed

    if current:
        chunks.append(current)

    return chunks


# =========================================================
# SAFE TELEGRAM RESPONSE
# =========================================================

async def safe_reply_text(
    status_msg,
    text: str,
    prefix: str = ""
):
    full_text = (
        f"{prefix}\n\n{text}"
        if prefix
        else text
    )

    chunks = split_text(
        full_text,
        MAX_TELEGRAM_MESSAGE_LENGTH
    )

    for index, chunk in enumerate(chunks):
        try:
            if index == 0:
                await status_msg.edit_text(chunk, parse_mode="Markdown")
            else:
                await status_msg.reply_text(chunk, parse_mode="Markdown")
        except Exception as e:
            logger.warning(f"Telegram message send failed: {e}")
            try:
                if index == 0:
                    await status_msg.edit_text(chunk, parse_mode=None)
                else:
                    await status_msg.reply_text(chunk, parse_mode=None)
            except Exception as final_error:
                logger.error(f"Final Telegram send error: {final_error}")


# =========================================================
# /START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not update.message:
        return

    await update.message.reply_text(
        "👑 **به بات تحلیل طلای امین خوش آمدید!** 🏆\n\n"
        "📈 سیستم هوشمند تحلیل تخصصی بازار جهانی طلا (XAUUSD)\n"
        "⚡️ مجهز به سیستم پیشرفته پرایس اکشن، اسمارت مانی (SMC) و NDS\n\n"
        "💎 عکس چارت معاملاتی خود را ارسال کنید تا گزارش تحلیل حرفه‌ای، سناریوهای دقیق و گریدبندی سیگنال (A+ تا C) را دریافت کنید.\n\n"
        "🔧 دستور تست اتصال: `/test`",
        parse_mode="Markdown"
    )


# =========================================================
# /TEST
# =========================================================

async def test_cmd(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not update.message:
        return

    status_msg = await update.message.reply_text(
        "⏳ در حال ارزیابی اتصال به موتور تحلیل هوشمند..."
    )

    try:
        text, used_model = await analyze_with_fallback(
            "Reply with exactly: OK"
        )

        await status_msg.edit_text(
            "✅ **سیستم بات تحلیل طلای امین کاملاً آماده و فعال است!** 🚀\n\n"
            f"🤖 **مدل فعال:** `{used_model}`\n"
            f"💬 **وضعیت:** ارتباط پایدار با سرورهای هوش مصنوعی برقرار است.",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.exception("Gemini test failed.")
        error_message = str(e)

        try:
            available_models = await asyncio.to_thread(_get_available_models_sync)
            if available_models:
                models_text = "\n".join(f"• `{model}`" for model in available_models[:20])
            else:
                models_text = "هیچ مدل Gemini قابل مشاهده‌ای پیدا نشد."
        except Exception:
            models_text = "امکان دریافت لیست مدل‌ها وجود نداشت."

        final_text = (
            "❌ اتصال به سیستم ناموفق بود.\n\n"
            f"🔴 خطا:\n`{error_message}`\n\n"
            "📋 مدل‌های قابل مشاهده برای API Key شما:\n"
            f"{models_text}"
        )

        await status_msg.edit_text(final_text, parse_mode="Markdown")


# =========================================================
# PHOTO HANDLER
# =========================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not update.message:
        return

    status_msg = await update.message.reply_text(
        "🔍 **بات تحلیل طلای امین** در حال پردازش تصویر، کالبدشکافی NDS و محاسبه ارزیابی ریسک است... ⏳"
    )

    try:
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        image = await asyncio.to_thread(process_image, bytes(photo_bytes))

        analysis, used_model = await analyze_with_fallback(
            SMC_PROMPT,
            image
        )

        prefix = (
            "👑 **گزارش تخصصی بات تحلیل طلای امین** 📊\n"
            f"🤖 موتور پردازشگر: `{used_model}`\n"
            "━━━━━━━━━━━━━━━━━━━━━━━"
        )

        await safe_reply_text(
            status_msg,
            analysis,
            prefix=prefix
        )

    except Exception as e:
        logger.exception("Image analysis failed.")
        error_text = f"❌ خطا در پردازش و تحلیل تصویر.\n\n🔴 جزئیات:\n`{str(e)}`"

        try:
            await status_msg.edit_text(error_text, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(error_text)


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):
    logger.exception(
        "Unhandled Telegram error:",
        exc_info=context.error
    )


# =========================================================
# MAIN
# =========================================================

def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not configured.")
        return

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not configured.")
        return

    if not client:
        logger.error("Gemini client could not be initialized.")
        return

    health_thread = Thread(
        target=start_health_check_server,
        daemon=True
    )
    health_thread.start()

    app = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("test", test_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_error_handler(error_handler)

    logger.info("====================================")
    logger.info("بات تحلیل طلای امین در حال راه‌اندازی...")
    logger.info("Gemini models:")
    for model in CANDIDATE_MODELS:
        logger.info(f" - {model}")
    logger.info("====================================")

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
