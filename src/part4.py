from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

thu_muc = Path(__file__).resolve().parent.parent

(thu_muc / "figures").mkdir(exist_ok=True)
(thu_muc / "tables").mkdir(exist_ok=True)

ten_cot = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
data = pd.read_csv(thu_muc / "data/raw/iris.data", header=None, names=ten_cot)

# X gồm 4 cột số đo; y là cột tên loài hoa.
ten_dac_trung = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = data[ten_dac_trung]
y = data["species"]

X_minmax = pd.DataFrame()
X_zscore = pd.DataFrame()
for cot in ten_dac_trung:
    nho_nhat = X[cot].min()
    lon_nhat = X[cot].max()
    trung_binh = X[cot].mean()
    do_lech_chuan = X[cot].std(ddof=0)

    X_minmax[cot] = (X[cot] - nho_nhat) / (lon_nhat - nho_nhat)
    X_zscore[cot] = (X[cot] - trung_binh) / do_lech_chuan

print("Mẫu 0 ban đầu:")
print(X.iloc[0])
print("\nMẫu 0 sau Min-Max:")
print(X_minmax.iloc[0].round(6))
print("\nMẫu 0 sau Z-score:")
print(X_zscore.iloc[0].round(6))
X_minmax.to_csv(thu_muc / "tables/minmax.csv", index=False)
X_zscore.to_csv(thu_muc / "tables/zscore.csv", index=False)

#Thống kê bốn đặc trưng trong ba trường hợp.
cac_bang = [X, X_minmax, X_zscore]
ten_phuong_phap = ["Original", "Min-Max", "Z-score"]
ket_qua = []
for i in range(3):
    bang = cac_bang[i]
    for cot in ten_dac_trung:
        ket_qua.append([ten_phuong_phap[i], cot, bang[cot].min(),
                        bang[cot].max(), bang[cot].mean(), bang[cot].std(ddof=0)])

thong_ke = pd.DataFrame(ket_qua, columns=["Method", "Feature", "Min", "Max", "Mean", "Std_ddof0"])
print("\nBảng so sánh trước và sau chuẩn hóa:")
print(thong_ke.round(6).to_string(index=False))
thong_ke.to_csv(thu_muc / "tables/normalization.csv", index=False)

# Vẽ cùng cặp tốt nhất petal_length và petal_width.
plt.figure(figsize=(16, 5))
for i in range(3):
    bang = cac_bang[i]
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(x=bang["petal_length"], y=bang["petal_width"], hue=y)
    if i == 0:
        plt.xlabel("petal_length (cm)")
        plt.ylabel("petal_width (cm)")
    else:
        plt.xlabel("petal_length")
        plt.ylabel("petal_width")
    plt.title(ten_phuong_phap[i])
plt.tight_layout()
plt.savefig(thu_muc / "figures/08_normalization_scatter.png", dpi=150)

# 4. Xem phân phối bằng histogram
hai_dac_trung = ["petal_length", "petal_width"]
plt.figure(figsize=(16, 9))
for i in range(2):
    cot = hai_dac_trung[i]
    for j in range(3):
        bang = cac_bang[j]
        plt.subplot(2, 3, i * 3 + j + 1)
        sns.histplot(x=bang[cot], hue=y, bins=15, multiple="layer", alpha=0.5)
        if j == 0:
            plt.xlabel(cot + " (cm)")
        else:
            plt.xlabel(cot)
        plt.ylabel("Count")
        plt.title(ten_phuong_phap[j] + " - " + cot)
plt.tight_layout()
plt.savefig(thu_muc / "figures/09_distributions.png", dpi=150)

plt.show()
