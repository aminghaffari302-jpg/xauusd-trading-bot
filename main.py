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

# حداکثر ضلع تصویر
MAX_IMAGE_DIMENSION = 1536

# محدودیت تلگرام
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
# مدل‌های استاندارد و فعال API گوگل در حال حاضر
# ترتیب: از سریع‌ترین و قدرتمندترین مدل تصویر به مدل‌های جایگزین

CANDIDATE_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
]


# =========================================================
# SMC PROMPT
# =========================================================

SMC_PROMPT = """
تو یک تحلیل‌گر حرفه‌ای تکنیکال و متخصص Smart Money Concepts (SMC)
هستی و باید تصویر چارت XAUUSD (طلا) را بررسی کنی.

هدف این است که فقط بر اساس اطلاعات قابل مشاهده در تصویر تحلیل کنی.

⚠️ قوانین بسیار مهم:

1. هیچ قیمت، Entry، SL یا TP را از خودت حدس نزن.
2. اگر قیمت‌ها یا ساختار چارت واضح نیست، صریحاً بگو اطلاعات کافی نیست.
3. اگر تایم‌فریم مشخص است، آن را اعلام کن.
4. اگر تایم‌فریم مشخص نیست، حدس نزن.
5. بین اطلاعات قابل مشاهده و برداشت تحلیلی تفاوت قائل شو.
6. تحلیل را قطعی و تضمینی معرفی نکن.
7. اگر شرایط ورود مناسب نیست، صریحاً بگو "فعلاً ورود مناسب نیست".
8. قبل از ارائه Entry/SL/TP، ابتدا ساختار بازار را بررسی کن.
9. فقط Order Block یا FVGهایی را مطرح کن که واقعاً در تصویر قابل تشخیص باشند.
10. اگر تصویر کیفیت کافی ندارد، به جای ساختن تحلیل، محدودیت تصویر را اعلام کن.

تحلیل را با ساختار زیر ارائه بده:

👑 تحلیل اختصاصی SMC | XAUUSD

📌 ۱. اطلاعات چارت
• تایم‌فریم:
• وضعیت قابل مشاهده قیمت:
• کیفیت تصویر:

📊 ۲. ساختار بازار
• روند فعلی:
• BOS / BMS:
• CHOCH:
• نقدینگی مهم:
• سقف‌ها و کف‌های مهم:

⚖️ ۳. نواحی مهم
• Order Block:
• Fair Value Gap:
• Liquidity:
• نواحی Supply / Demand:

🎯 ۴. سناریوی معاملاتی

سناریوی اصلی:
• جهت: BUY / SELL / NO TRADE
• Entry:
• SL:
• TP1:
• TP2:
• TP3:
• R:R:

سناریوی جایگزین:
• شرط فعال شدن:
• Entry:
• SL:
• TP:

🛡️ ۵. مدیریت ریسک
• نقطه invalidation:
• چه زمانی نباید وارد معامله شد؟
• ریسک پیشنهادی:
• نکته مهم:

🔎 ۶. جمع‌بندی
در چند خط بگو در حال حاضر مهم‌ترین سناریوی قابل مشاهده چیست.

اگر اطلاعات کافی برای تعیین دقیق Entry، SL یا TP وجود ندارد،
به‌جای حدس زدن بنویس:
"برای تعیین عدد دقیق، اطلاعات چارت کافی نیست."
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

        logger.info(
            f"Health check server running on port {PORT}"
        )

        server.serve_forever()

    except Exception as e:
        logger.error(
            f"Health check server failed: {e}"
        )


# =========================================================
# IMAGE PROCESSING
# =========================================================

def process_image(image_bytes: bytes) -> Image.Image:
    """
    Resize image while keeping aspect ratio.
    Converts image to RGB.
    """

    with Image.open(io.BytesIO(image_bytes)) as img:

        # اصلاح جهت عکس‌های موبایل
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        img.thumbnail(
            (
                MAX_IMAGE_DIMENSION,
                MAX_IMAGE_DIMENSION
            ),
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
        raise RuntimeError(
            "Gemini client is not initialized."
        )

    if image is not None:
        contents = [image, prompt]
    else:
        contents = prompt

    response = client.models.generate_content(
        model=model_name,
        contents=contents
    )

    if not response:
        return ""

    text = getattr(response, "text", None)

    if not text:
        return ""

    return text.strip()


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
                await status_msg.edit_text(chunk)
            else:
                await status_msg.reply_text(chunk)
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
        "سلام 👋\n\n"
        "🤖 ربات تحلیل چارت طلا (XAUUSD) فعال است.\n\n"
        "📷 عکس چارت را ارسال کنید تا آن را با "
        "استفاده از SMC تحلیل کنم.\n\n"
        "🧪 برای تست اتصال Gemini:\n"
        "/test"
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
        "⏳ در حال تست اتصال به Gemini..."
    )

    try:

        text, used_model = await analyze_with_fallback(
            "Reply with exactly: OK"
        )

        await status_msg.edit_text(
            "✅ اتصال Gemini موفق بود!\n\n"
            f"🤖 مدل فعال:\n`{used_model}`\n\n"
            f"💬 پاسخ:\n{text}",
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
            "❌ اتصال به Gemini ناموفق بود.\n\n"
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
        "⏳ عکس دریافت شد.\n"
        "🔍 در حال تحلیل چارت..."
    )

    try:

        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        logger.info(f"Downloaded image: {len(photo_bytes)} bytes")

        image = await asyncio.to_thread(process_image, bytes(photo_bytes))

        logger.info(f"Processed image size: {image.size}")

        analysis, used_model = await analyze_with_fallback(
            SMC_PROMPT,
            image
        )

        prefix = (
            "📊 تحلیل تکنیکال SMC | XAUUSD\n"
            f"🤖 مدل: `{used_model}`"
        )

        await safe_reply_text(
            status_msg,
            analysis,
            prefix=prefix
        )

    except Exception as e:

        logger.exception("Image analysis failed.")
        error_text = f"❌ خطا در تحلیل تصویر.\n\n🔴 جزئیات:\n`{str(e)}`"

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
    logger.info("Telegram bot is starting...")
    logger.info("Gemini models:")
    for model in CANDIDATE_MODELS:
        logger.info(f" - {model}")
    logger.info("====================================")

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
