# Chạy lần lượt 4 phần. Có thể chạy riêng từng part nếu đang học.
from pathlib import Path
import subprocess
import sys

# sys.executable là Python đang chạy file này.
# subprocess.run chạy một file khác; check=True dừng nếu file đó bị lỗi.
thu_muc_code = Path(__file__).resolve().parent
cac_file = ["part1.py", "part2.py", "part3.py", "part4.py"]

for ten_file in cac_file:
    print("Đang chạy:", ten_file, flush=True)
    print("Đóng các cửa sổ biểu đồ để chuyển sang phần tiếp theo.", flush=True)
    subprocess.run([sys.executable, str(thu_muc_code / ten_file)], check=True)
