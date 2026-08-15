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

# --- Bot Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👑 **سلام! به ربات سیگنال طلای امین خوش اومدید.**\n\n"
        "📈 عکس چارت طلای مد نظرتون رو بفرستید تا براتون تحلیل کنم.",
        parse_mode="Markdown"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_message = await update.message.reply_text("🔎 **در حال اسکن چارت و کالبدشکافی ساختار بازار توسط مدل هوشمند...**")
    file_path = "chart.jpg"
    try:
        # Download photo
        photo_file = await update.message.photo[-1].get_file()
        await photo_file.download_to_drive(file_path)

        # Open image with Pillow
        img = Image.open(file_path)

        # Retry logic for handling temporary 503 server overload
        max_retries = 3
        response = None
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-flash-latest',
                    contents=[img, ADVANCED_PA_PROMPT]
                )
                break
            except Exception as api_err:
                if "503" in str(api_err) and attempt < max_retries - 1:
                    await status_message.edit_text(f"⏳ **سرور در حال پردازش سنگین است. تلاش مجدد ({attempt + 1}/{max_retries})...**")
                    await asyncio.sleep(3)
                else:
                    raise api_err
        
        # Clean up local image file
        if os.path.exists(file_path):
            os.remove(file_path)

        if response and response.text:
            try:
                await status_message.edit_text(response.text, parse_mode="Markdown")
            except Exception:
                await status_message.edit_text(response.text)
        else:
            await status_message.edit_text("❌ پاسخی از مدل دریافت نشد. لطفاً مجدداً تلاش کنید.")

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        await status_message.edit_text(f"⚠️ **خطا در تحلیل چارت:**\n\n{str(e)}")

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

    print("Bot is starting with customized welcome message...")
    app.run_polling()

if __name__ == "__main__":
    main()
