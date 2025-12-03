import serial
import time
from datetime import datetime

# === 設定區 ===
SERIAL_PORT = 'COM5'     # 記得改成你的 Port
BAUD_RATE = 115200
FILE_NAME = 'data_60s.txt'

# ⏳ 想要狩獵多久？（單位：秒）
# 如果設為 None，我就會一直抓一直抓，直到你按 Ctrl+C 強制把我的頭按下去為止
HUNTING_DURATION = 60  

def start_hunting():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"喵嗚～潛入 {SERIAL_PORT}！")
        
        if HUNTING_DURATION:
            print(f"這次只玩 {HUNTING_DURATION} 秒喔，計時開始！")
        else:
            print("無限暢飲模式！(按 Ctrl+C 結束)")

        start_time = time.time() # 記下開始的時間點

        with open(FILE_NAME, 'a', encoding='utf-8') as f:
            while True:
                # 1. 檢查有沒有超時，時間到了就收工回家睡覺
                if HUNTING_DURATION and (time.time() - start_time > HUNTING_DURATION):
                    print(f"\n時間到啦！{HUNTING_DURATION} 秒結束。收工～（伸懶腰）")
                    break

                # 2. 看看有沒有資料進來
                if ser.in_waiting > 0:
                    raw_data = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if raw_data:
                        # 3. 抓取現在的時間，做成標籤
                        #timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # 精確到毫秒
                        
                        # 組合起來： [時間] 資料
                        log_entry = f"{raw_data}"
                        
                        print(log_entry) # 螢幕上也叫一聲給你看
                        f.write(log_entry + "\n")
                        f.flush()
                        
    except KeyboardInterrupt:
        print("\n中途被你打斷了... 唔。")
    except Exception as e:
        print(f"\n嘶！出錯了：{e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("通道關閉。")

if __name__ == "__main__":
    start_hunting()