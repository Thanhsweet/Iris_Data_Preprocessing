# PHẦN 2: CHỌN ĐẶC TRƯNG
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

# 1. Liệt kê rõ 6 cặp, thay vì dùng công cụ sinh tổ hợp.
cac_cap = [
    ["sepal_length", "sepal_width"],
    ["sepal_length", "petal_length"],
    ["sepal_length", "petal_width"],
    ["sepal_width", "petal_length"],
    ["sepal_width", "petal_width"],
    ["petal_length", "petal_width"]
]

plt.figure(figsize=(16, 10))
for i in range(6):
    cot_x = cac_cap[i][0]
    cot_y = cac_cap[i][1]
    plt.subplot(2, 3, i + 1)  # 2 hàng, 3 cột, ô thứ i+1.
    sns.scatterplot(data=data, x=cot_x, y=cot_y, hue="species")
    plt.xlabel(cot_x + " (cm)")
    plt.ylabel(cot_y + " (cm)")
    plt.title(cot_x + " - " + cot_y)
plt.tight_layout()
plt.savefig(thu_muc / "figures/03_scatter_2d.png", dpi=150)

# 2. Liệt kê rõ 4 bộ ba đặc trưng.
cac_bo_ba = [
    ["sepal_length", "sepal_width", "petal_length"],
    ["sepal_length", "sepal_width", "petal_width"],
    ["sepal_length", "petal_length", "petal_width"],
    ["sepal_width", "petal_length", "petal_width"]
]
cac_loai = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
mau_sac = ["tab:blue", "tab:orange", "tab:green"]

hinh = plt.figure(figsize=(16, 12))
for i in range(4):
    cot_x = cac_bo_ba[i][0]
    cot_y = cac_bo_ba[i][1]
    cot_z = cac_bo_ba[i][2]
    truc = hinh.add_subplot(2, 2, i + 1, projection="3d")
    for j in range(3):
        # Lọc những dòng thuộc cùng một loài hoa.
        du_lieu_loai = data[data["species"] == cac_loai[j]]
        truc.scatter(du_lieu_loai[cot_x], du_lieu_loai[cot_y],
                     du_lieu_loai[cot_z], color=mau_sac[j], label=cac_loai[j])
    truc.set_xlabel(cot_x + " (cm)")
    truc.set_ylabel(cot_y + " (cm)")
    truc.set_zlabel(cot_z + " (cm)")
    truc.set_title("Combination " + str(i + 1))
    truc.legend()
plt.tight_layout()
plt.savefig(thu_muc / "figures/04_scatter_3d.png", dpi=150, bbox_inches="tight")

# 3. Tính Fisher Score = độ phân tán giữa lớp / độ phân tán trong lớp.
# Tính lần lượt từng đặc trưng, rồi từng lớp bằng vòng lặp for.
ket_qua = []
for cot in ten_dac_trung:
    trung_binh_chung = data[cot].mean()
    giua_lop = 0
    trong_lop = 0

    for loai in cac_loai:
        nhom = data[data["species"] == loai]
        so_mau = len(nhom)
        trung_binh_lop = nhom[cot].mean()
        phuong_sai_lop = nhom[cot].var(ddof=0)

        # B = tổng n_c * (mean_c - mean_chung)^2.
        giua_lop = giua_lop + so_mau * (trung_binh_lop - trung_binh_chung) ** 2
        # W = tổng n_c * variance_c; variance_c dùng mẫu số n_c.
        trong_lop = trong_lop + so_mau * phuong_sai_lop

    diem_fisher = giua_lop / trong_lop  # Cả 4 cột Iris đều có trong_lop > 0.
    ket_qua.append([cot, giua_lop, trong_lop, diem_fisher])

bang_fisher = pd.DataFrame(ket_qua, columns=["Feature", "B", "W", "Fisher"])
bang_fisher = bang_fisher.sort_values(by="Fisher", ascending=False)
print("Bảng xếp hạng Fisher:")
print(bang_fisher.round(6).to_string(index=False))
bang_fisher.to_csv(thu_muc / "tables/fisher.csv", index=False)
print("\nHai đặc trưng có Fisher cao nhất:")
print(bang_fisher["Feature"].head(2).to_list())
# Đây là Fisher B/W, không phải ANOVA F của hàm f_classif.
# Với Iris, ANOVA F = 73.5 * Fisher B/W.

# 4. Vẽ biểu đồ cột xếp hạng.
plt.figure(figsize=(9, 5))
plt.bar(bang_fisher["Feature"], bang_fisher["Fisher"], color="teal")
plt.xlabel("Feature")
plt.ylabel("Fisher score (B/W)")
plt.title("Fisher feature ranking")
plt.tight_layout()
plt.savefig(thu_muc / "figures/05_fisher.png", dpi=150)
plt.show()
