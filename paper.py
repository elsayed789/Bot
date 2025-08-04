# ✅ نسخة المحاكاة اللحظية المتطابقة تمامًا مع الباك تست

import pandas as pd
import ta
import joblib
import json
import time
import threading
from datetime import datetime, timedelta
from binance.client import Client
from websocket import WebSocketApp

# ===== إعدادات Binance =====
API_KEY = "rtPG4RdT0OqZJuz2D8oZntoqCZJc4XnCFL7rKpVkCScQrYwpppHtREZSmjo62j2P"
API_SECRET = "TXG3Tf8ylyMGIMzl6JNutXFnCWnVb9eEOKwZ5jXasLPRX8BJkBZY2wygS75pGZ8o"
symbol = "SUIUSDT"
symbol_ws = "suiusdt"
interval = Client.KLINE_INTERVAL_15MINUTE
LOOKBACK_LIMIT = 1000

# ===== إعدادات البوت =====
TAKE_PROFIT = 0.03
STOP_LOSS = -0.015
CONFIDENCE_THRESHOLD = 0.6
FEE = 0.001
SLIPPAGE = 0.0012
START_BALANCE = 1000
TRADE_PORTION = 0.5
MAX_BARS_IN_TRADE = 10000

# ===== تحميل الموديل =====
model, important_features = joblib.load("trained_model.pkl")

# ===== حالة التداول =====
client = Client(API_KEY, API_SECRET)
balance = START_BALANCE
in_position = False
entry_price = 0
token_amount = 0
entry_time = None
results = []
all_candles = []

print("✅ بدء المحاكاة اللحظية... تطابق تام مع الباك تست")

def on_new_candle():
    global balance, in_position, entry_price, token_amount, entry_time, results, all_candles

    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=1)
        k = klines[-1]
        candle = {
            "timestamp": datetime.fromtimestamp(k[0] / 1000),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        }
        all_candles.append(candle)
        df = pd.DataFrame(all_candles)

        # حساب المؤشرات الفنية
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bollinger_mavg'] = bb.bollinger_mavg()
        df['bollinger_h'] = bb.bollinger_hband()
        df['bollinger_l'] = bb.bollinger_lband()
        df['ema'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
        df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()

        df.dropna(inplace=True)
        if len(df) < 10:
            return

        row = df.iloc[-1]
        features = df[important_features].iloc[-1:]

        current_price = row['open']
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][1]

        # تسجيل البيانات في ملف live_data.csv كل مرة
        row_to_save = row.copy()
        row_to_save['signal'] = prediction
        pd.DataFrame([row_to_save]).to_csv("live_data.csv", mode='a', index=False, header=not pd.io.common.file_exists("live_data.csv"))

        if not in_position:
            if confidence >= CONFIDENCE_THRESHOLD:
                entry_price = current_price * (1 + SLIPPAGE)
                entry_time = row['timestamp']
                trade_value = balance * TRADE_PORTION
                token_amount = trade_value / entry_price
                in_position = True
                print(f"📈 شراء @ {entry_price:.4f} | الوقت: {entry_time} | الثقة: {confidence:.3f}")

        else:
            change = (current_price - entry_price) / entry_price
            bars_in_trade = (row['timestamp'] - entry_time) / timedelta(minutes=15)

            # شروط الخروج مطابقة تمامًا للباك تست
            if change >= TAKE_PROFIT:
                reason = "take_profit"
            elif change <= STOP_LOSS:
                reason = "stop_loss"
            elif prediction == 0 and confidence < CONFIDENCE_THRESHOLD:
                reason = "reverse"
            elif bars_in_trade >= MAX_BARS_IN_TRADE:
                reason = "timeout"
            else:
                return

            exit_price = current_price * (1 - SLIPPAGE)
            net_change = exit_price - entry_price
            gross_result = net_change * token_amount
            fees = (entry_price + exit_price) * FEE * token_amount
            net_result = gross_result - fees
            balance += net_result

            results.append({
                "entry_time": entry_time,
                "exit_time": row['timestamp'],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "result": reason,
                "profit": net_result if net_result > 0 else 0,
                "loss": -net_result if net_result < 0 else 0,
                "net_profit_per_trade": net_result,
                "balance": balance
            })

            print(f"🔁 {reason.upper()} | الخروج @ {exit_price:.4f} | الصافي: {net_result:.4f} | الرصيد: {balance:.2f}")
            in_position = False

            pd.DataFrame(results).to_csv("live_simulation_log.csv", index=False)

    except Exception as e:
        print(f"❗ خطأ أثناء معالجة الشمعة: {e}")

# WebSocket

def on_message(ws, message):
    data = json.loads(message)
    if data['k']['x']:
        print("🕯️ شمعة جديدة مكتملة...")
        on_new_candle()

def on_error(ws, error):
    print(f"❗ خطأ WebSocket: {error}")

def on_close(ws, *_):
    print("🔌 WebSocket مغلق، إعادة الاتصال بعد 10 ثواني...")
    time.sleep(10)
    start_websocket()

def start_websocket():
    socket_url = f"wss://stream.binance.com:9443/ws/{symbol_ws}@kline_15m"
    ws = WebSocketApp(socket_url, on_message=on_message, on_error=on_error, on_close=on_close)
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()

start_websocket()

while True:
    time.sleep(1)    