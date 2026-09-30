# PHẦN 1: TÌM HIỂU DỮ LIỆU IRIS
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Xác định thư mục dự án từ vị trí file .py.
# Ví dụ: Iris_Code_De_Hieu/src/part1.py -> Iris_Code_De_Hieu
thu_muc = Path(__file__).resolve().parent.parent

# Tạo nơi lưu hình và bảng kết quả nếu chưa có.
(thu_muc / "figures").mkdir(exist_ok=True)
(thu_muc / "tables").mkdir(exist_ok=True)

# File iris.data không có dòng tiêu đề nên tự đặt tên cột.
ten_cot = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
data = pd.read_csv(thu_muc / "data/raw/iris.data", header=None, names=ten_cot)

# X gồm 4 cột số đo; y là cột tên loài hoa.
ten_dac_trung = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = data[ten_dac_trung]
y = data["species"]
print("x của mẫu đầu tiên:")
print(X.iloc[0])

print("y của mẫu đầu tiên:")
print(y.iloc[0])
# 1. Xem dữ liệu và kiểm tra số lượng.
print("Toàn bộ dữ liệu:")
print(data.to_string())
print("\nSố mẫu:", len(data))
print("Số đặc trưng:", len(ten_dac_trung))
print("Số lớp:", y.nunique())
print("\nSố mẫu từng lớp:")
print(y.value_counts())
print("\nKiểu dữ liệu trong Python:")
print(data.dtypes)
print("\nSố giá trị bị thiếu:")
print(data.isnull().sum())

# Các số đo: định lượng liên tục, thang đo tỷ lệ, đơn vị cm.
# species: thuộc tính định tính, không có thứ tự.
print("\nMa trận X (đầy đủ):")
print(X.to_string())
print("\nVector nhãn y (đầy đủ):")
print(y.to_string())
X.to_csv(thu_muc / "tables/X.csv", index=False)
y.to_csv(thu_muc / "tables/y.csv", index=False)
data.to_csv(thu_muc / "tables/training_set.csv", index_label="sample_index")

# 2. Tính riêng từng thống kê để dễ theo dõi.
thong_ke = pd.DataFrame()
thong_ke["Min"] = X.min()
thong_ke["Max"] = X.max()
thong_ke["Mean"] = X.mean()
thong_ke["Std_mau"] = X.std(ddof=1)
thong_ke["Std_ddof0"] = X.std(ddof=0)
# ddof=1: chia cho n-1; ddof=0: chia cho n trước khi lấy căn
print("\nBảng thống kê:")
print(thong_ke.round(6))
thong_ke.to_csv(thu_muc / "tables/thong_ke.csv")

# 3. Tính tương quan Pearson cho bốn cột số.
tuong_quan = X.corr(method="pearson")
print("\nMa trận Pearson:")
print(tuong_quan.round(6))
tuong_quan.to_csv(thu_muc / "tables/pearson.csv")

# 4. Vẽ heatmap: màu biểu thị mức độ tương quan.
plt.figure(figsize=(9, 7))
sns.heatmap(tuong_quan, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Pearson correlation - Iris")
plt.tight_layout()
plt.savefig(thu_muc / "figures/01_heatmap.png", dpi=150)

# 5. Vẽ pairplot: mỗi màu là một loài hoa.
sns.pairplot(data, vars=ten_dac_trung, hue="species", diag_kind="hist")
plt.suptitle("Iris feature pairs - measurements in cm", y=1.02)
plt.savefig(thu_muc / "figures/02_pairplot.png", dpi=150, bbox_inches="tight")

# Hiển thị các hình. Đóng cửa sổ hình khi muốn kết thúc chương trình.
plt.show()
