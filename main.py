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

# --- Config Gemini API ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# --- Advanced Institutional Price Action Prompt ---
ADVANCED_PA_PROMPT = (
    "You are a Lead Institutional Technical Analyst specializing in Price Action, Smart Money Concepts (SMC), and RTM for XAUUSD (Gold). "
    "Analyze this chart image with surgical precision and structural depth. Produce a clean, highly structured report entirely in Persian (Farsi).\n\n"
    "Use a professional, sharp, and visually clear tone with effective structural emojis (such as 👑, 📊, 🎯, ⚖️, 🛑, 🚀, ⚠️, 🟩, 🟦, 🟧).\n\n"
    "Follow this EXACT response structure in Persian:\n\n"
    "👑 **اتاق تحلیل اختصاصی طلا | XAUUSD VIP Analysis**\n"
    "---\n\n"
    "📊 **۱. کالبدشکافی ساختار بازار (Market Structure & SMC)**\n"
    "• **روند و مومنتوم:** [صعودی / نزولی / رنج] + بررسی توان خریداران و فروشندگان.\n"
    "• **تغییر ماهیت و شکست ساختار (CHOCH / BMS):** مشخص کردن آخرین شکست‌های ساختاری.\n"
    "• **نواحی نقدینگی و گپ‌ها (Liquidity & FVG):** شناسایی استخر نقدینگی (BSL/SSL) و گپ‌های ارزش منصفانه (Fair Value Gaps).\n"
    "• **بلاک‌های سفارش فعال (Order Blocks):** مشخص کردن دقیق زون‌های تقاضا (Demand) یا عرضه (Supply).\n\n"
    "---\n\n"
    "⚖️ **۲. سطوح و زون‌های کلیدی (Key Zones & Levels)**\n"
    "🔴 **زون‌های عرضه / مقاومت:** [اعداد دقیق visible روی چارت]\n"
    "🟢 **زون‌های تقاضا / حمایت:** [اعداد دقیق visible روی چارت]\n"
    "🌐 **محدوده تعادلی (Equilibrium / P&D):** وضعیت قیمت در زون‌های G&D (Premium vs Discount).\n\n"
    "---\n\n"
    "🎯 **۳. سناریوها و درجه‌بندی سیگنال‌های معاملاتی (Trading Setups)**\n\n"
    "🟩 **سناریو A+ (سیگنال کم‌ریسک / هم‌جهت با روند اصلی):**\n"
    "• **نوع پوزیشن:** [Long / Short]\n"
    "• **محدوده ورود (EP):** [عدد دقیق یا زون مشخص]\n"
    "• **حد ضرر (SL):** [عدد دقیق + دلیل ساختاری]\n"
    "• **حد سود اول (TP1):** [عدد دقیق]\n"
    "• **حد سود دوم (TP2):** [عدد دقیق]\n"
    "• **نسبت ریسک به بهای پرداختی (R:R):** [مثال: 1:3]\n\n"
    "🟦 **سناریو B (سیگنال ریسک متوسط / پولبک و تأیید مجدد):**\n"
    "• **نوع پوزیشن:** [Long / Short]\n"
    "• **محدوده ورود (EP):** [عدد یا محدوده]\n"
    "• **حد ضرر (SL):** [عدد دقیق]\n"
    "• **حد سود (TP):** [اعداد دقیق]\n\n"
    "🟧 **سناریو C (سیگنال پرریسک / خلاف جهت روند یا اسکلپ سریع):**\n"
    "• **شرایط ورود:** [توضیح کوتاه]\n"
    "• **محدوده ورود / SL / TP:** [مشخصات دقیق]\n\n"
    "---\n\n"
    "🛡️ **۴. مدیریت ریسک و توصیه استراتژیک (Trading Plan & Warning)**\n"
    "• **توصیه حجم:** [مثال: حداکثر ۱ الی ۲ درصد ریسک کل حساب]\n"
    "• **تأییدیه‌های لازم (Confirmations):** [نوع کندل بازگشتی یا شکست تایم‌پایین مورد نیاز]\n"
    "• **هشدار مهم:** [توضیح مختصر در مورد اخبارهای پیش‌رو یا نوسانات طلا]\n"
)

# --- Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👑 **سلام! به ربات سیگنال طلای امین خوش اومدید.**\n\n"
        "📈 عکس چارت طلای مد نظرتون رو بفرستید تا براتون تحلیل کنم.",
        parse_mode="Markdown"
    )

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 **دریافت جدیدترین تحلیل طلا**\n\n"
        "لطفاً تصویر چارت مورد نظر خود را ارسال کنید تا تحلیل SMC اختصاصی آن تولید شود.",
        parse_mode="Markdown"
    )

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎯 **دریافت وضعیت سیگنال‌های فعلی**\n\n"
        "سیگنال‌های معاملاتی فعال بر روی چارت‌های ارسالی پردازش می‌شوند. عکس چارت جدید خود را بفرستید.",
        parse_mode="Markdown"
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📊 **وضعیت اتصال به متاتریدر (MetaTrader):**\n\n"
        "بزودی در بروزرسانی‌های آینده اضافه میشه😊",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ **راهنمای استفاده از ربات:**\n\n"
        "۱. یک عکس واضح از چارت طلا (XAUUSD) ارسال کنید.\n"
        "۲. هوش مصنوعی ساختار بازار، زون‌ها و سناریوهای A+، B و C را محاسبه می‌کند.\n"
        "۳. سیگنال‌های ورودی به همراه TP و SL را دریافت کنید.",
        parse_mode="Markdown"
    )

# --- Photo Handler ---
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_message = await update.message.reply_text("📥 **تصویر دریافت شد؛ در حال پردازش اولیه...**")
    file_path = f"chart_{update.message.message_id}.jpg"
    
    try:
        photo_file = await update.message.photo[-1].get_file()
        await photo_file.download_to_drive(file_path)
        
        img = Image.open(file_path)
        img.thumbnail((1024, 1024))

        await status_message.edit_text("🧠 **در حال کالبدشکافی چارت و شناسایی زون‌های SMC...**")

        max_attempts = 2
        response = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                # استفاده از کلاینت ناهمگام ناتیو (client.aio) بدون ریسک گیر کردن ترد
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model='gemini-flash-latest',
                        contents=[img, ADVANCED_PA_PROMPT]
                    ),
                    timeout=30.0
                )
                if response and response.text:
                    break
            except Exception:
                if attempt < max_attempts:
                    await status_message.edit_text("⏳ **ترافیک سرور بالا است؛ در حال بازخوانی اطلاعات...**")
                    await asyncio.sleep(1.5)
        
        if os.path.exists(file_path):
            os.remove(file_path)

        if response and response.text:
            try:
                await status_message.edit_text(response.text, parse_mode="Markdown")
            except Exception:
                await status_message.edit_text(response.text)
        else:
            await status_message.edit_text("⚠️ **پاسخی دریافت نشد. لطفاً مجدداً تصویر چارت را ارسال کنید.**")

    except Exception:
        if os.path.exists(file_path):
            os.remove(file_path)
        await status_message.edit_text("⚠️ **پاسخ‌دهی سرور بیش از حد طولانی شد.**\nلطفاً چند ثانیه بعد مجدداً تصویر را ارسال کنید.")

def main():
    threading.Thread(target=run_http_server, daemon=True).start()

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN environment variable is missing.")
        return

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("help", help_command))
    
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Bot is running with native async client (client.aio)...")
    app.run_polling()

if __name__ == "__main__":
    main()
