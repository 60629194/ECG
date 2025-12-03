import serial
import time

strPort = 'COM5'  # 記得改成你的 Port
baud = 115200

try:
    ser = serial.Serial(strPort, baud)
    ser.flush()
    print(f"喵～正在連線 {strPort}，準備開始測速...")
    print("請稍等 10 秒鐘，讓我聽聽它的心跳節奏...")
    
    # 清空緩衝區，確保從現在開始算
    ser.reset_input_buffer()
    
    start_time = time.time()
    count = 0
    duration = 10.0 # 測試 10 秒
    
    while True:
        # 盡量讀
        if ser.in_waiting:
            try:
                ser.readline()
                count += 1
            except:
                pass
        
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

    real_fs = count / elapsed
    print("-" * 30)
    print(f"總共收到資料點數: {count} 筆")
    print(f"花費時間: {elapsed:.4f} 秒")
    print(f"★ 真實取樣頻率 (FS) 約為: {real_fs:.2f} Hz")
    print("-" * 30)
    print("請把這個數字填回主程式的 FS 設定裡！")
    
    ser.close()

except Exception as e:
    print(f"嗚... 發生錯誤: {e}")