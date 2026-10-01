from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

thu_muc = Path(__file__).resolve().parent.parent

(thu_muc / "figures").mkdir(exist_ok=True)
(thu_muc / "tables").mkdir(exist_ok=True)

ten_cot = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
data = pd.read_csv(thu_muc / "data/raw/iris.data", header=None, names=ten_cot)

# X gồm 4 cột số đo; y là cột tên loài hoa.
ten_dac_trung = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = data[ten_dac_trung]
y = data["species"]

# Chuẩn hóa trước khi làm PCA.

bo_chuan_hoa = StandardScaler()
X_chuan_hoa = bo_chuan_hoa.fit_transform(X)
X_chuan_hoa = pd.DataFrame(X_chuan_hoa, columns=ten_dac_trung)
print("Dữ liệu sau chuẩn hóa (5 dòng đầu):")
print(X_chuan_hoa.head().round(6))
X_chuan_hoa.to_csv(thu_muc / "tables/X_chuan_hoa.csv", index=False)

# Tính tất cả 4 thành phần để xem tỷ lệ phương sai.
pca = PCA(n_components=4, svd_solver="full")
X_pca = pca.fit_transform(X_chuan_hoa)
ty_le = pca.explained_variance_ratio_

# Cộng dồn từng tỷ lệ
tich_luy = []
tong = 0
for gia_tri in ty_le:
    tong = tong + gia_tri
    tich_luy.append(tong)

bang_pca = pd.DataFrame()
bang_pca["PC"] = ["PC1", "PC2", "PC3", "PC4"]
bang_pca["Eigenvalue"] = pca.explained_variance_
bang_pca["Variance_ratio"] = ty_le
bang_pca["Cumulative_ratio"] = tich_luy
print("\nBảng phương sai PCA:")
print(bang_pca.round(6).to_string(index=False))
bang_pca.to_csv(thu_muc / "tables/pca_variance.csv", index=False)

# Tìm số PC ít nhất để giữ từ 95% phương sai trở lên.
for i in range(4):
    if tich_luy[i] >= 0.95:
        so_chieu = i + 1
        break 
print("\nSố chiều cần giữ:", so_chieu)
print("Phương sai giữ lại (%):", round(tich_luy[so_chieu - 1] * 100, 4))

# components_: mỗi hàng là trọng số của một PC.
trong_so = pd.DataFrame(pca.components_,
                       index=["PC1", "PC2", "PC3", "PC4"], columns=ten_dac_trung)
print("\nTrọng số các PC:")
print(trong_so.round(6))
trong_so.to_csv(thu_muc / "tables/pca_weights.csv")

du_lieu_giam_chieu = X_pca[:, :so_chieu]
pd.DataFrame(du_lieu_giam_chieu).to_csv(thu_muc / "tables/pca_reduced.csv", index=False)

# Vẽ PC1-PC2
bang_diem = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3", "PC4"])
bang_diem["species"] = y
bang_diem.to_csv(thu_muc / "tables/pca_scores.csv", index=False)
print("\nTọa độ PCA của mẫu đầu tiên:")
print(bang_diem.iloc[0])

# Vẽ phương sai riêng và phương sai tích lũy.
plt.figure(figsize=(8, 5))
plt.bar([1, 2, 3, 4], ty_le, label="Individual variance")
plt.plot([1, 2, 3, 4], tich_luy, "o-", color="orange", label="Cumulative variance")
plt.axhline(y=0.95, color="red", linestyle="--", label="95% threshold")
plt.xticks([1, 2, 3, 4])
plt.xlabel("Number of components")
plt.ylabel("Variance ratio (0 to 1)")
plt.title("PCA explained variance")
plt.legend()
plt.tight_layout()
plt.savefig(thu_muc / "figures/06_pca_variance.png", dpi=150)

# Vẽ dữ liệu trong không gian hai chiều
plt.figure(figsize=(9, 6))
sns.scatterplot(data=bang_diem, x="PC1", y="PC2", hue="species")
plt.xlabel("PC1 (" + str(round(ty_le[0] * 100, 2)) + "%)")
plt.ylabel("PC2 (" + str(round(ty_le[1] * 100, 2)) + "%)")
plt.title("Iris after PCA")
plt.tight_layout()
plt.savefig(thu_muc / "figures/07_pca_scatter.png", dpi=150)
plt.show()
