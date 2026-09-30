# Iris — Data Preprocessing

Mở **report/Bao_cao_Iris.html** để đọc bản báo cáo hoàn chỉnh offline, hoặc notebook trong Jupyter. Báo cáo có đủ lý thuyết, tính tay, code, bảng thực thi và nhận xét cho cả bốn phần.

| Thư mục / tệp | Nội dung |
|---|---|
|report/|Báo cáo kỹ thuật Markdown và HTML có ảnh nhúng, công thức MathML|
|notebooks/|Notebook với nội dung báo cáo, code và hình nhúng|
|src/|Bốn script độc lập và run_all.py|
|data/raw/|Đề PDF và toàn bộ dữ liệu gốc|
|tables/|Các bảng CSV, X, y, điểm PCA và dữ liệu chuẩn hóa|
|figures/|Chín hình PNG chất lượng cao|
|environment.json|Phiên bản Python và các thư viện đã dùng|
|requirements-lock.txt|Phiên bản thư viện cố định của lần thực thi|
|data_manifest.json|SHA-256 dữ liệu nguồn|

## Chạy lại

```bash
python -m pip install -r requirements.txt
python src/run_all.py
```

Có thể chạy riêng `python src/part1.py` đến `part4.py`. Script chạy được từ bất kỳ working directory nào; notebook cần đặt nguyên trong thư mục dự án. Kết quả ghi đè vào tables và figures. Các báo cáo và markdown kết quả trong notebook là bản chụp của lần chạy đã giao, không tự cập nhật nếu thay dữ liệu.

Dùng iris.data trong toàn bộ bài; hai mẫu khác biệt so với bezdekIris.data được ghi rõ trong báo cáo. Std mẫu ddof=1 dùng mô tả; StandardScaler dùng ddof=0. Giữ 2 PC đạt 95,8010% phương sai. Fisher chọn petal_length và petal_width.
