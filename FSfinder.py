import serial
import time
import numpy as np

PORT = "COM5"
BAUD = 115200

def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)

    timestamps = []

    print("Measuring sampling frequency... (Ctrl+C to stop)")

    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode(errors="ignore").strip()

                # try to parse an integer/float
                try:
                    _ = float(line)
                except:
                    continue  # skip invalid messages

                # record timestamp
                now = time.time()
                timestamps.append(now)

                # limit buffer size
                if len(timestamps) > 2000:
                    timestamps = timestamps[-2000:]

                # compute frequency only if enough samples
                if len(timestamps) > 10:
                    intervals = np.diff(timestamps)
                    avg_interval = np.mean(intervals[-100:])  # last 100 intervals
                    fs = 1.0 / avg_interval
                    print(f"Sampling frequency: {fs:.2f} Hz", end="\r")

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        ser.close()


if __name__ == "__main__":
    main()
