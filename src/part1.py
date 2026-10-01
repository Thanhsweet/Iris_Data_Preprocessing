from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


thu_muc = Path(__file__).resolve().parent.parent
(thu_muc / "figures").mkdir(exist_ok=True)
(thu_muc / "tables").mkdir(exist_ok=True)

ten_cot = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
data = pd.read_csv(thu_muc / "data/raw/iris.data", header=None, names=ten_cot)

ten_dac_trung = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = data[ten_dac_trung]
y = data["species"]
print("x của mẫu đầu tiên:")
print(X.iloc[0])
print("y của mẫu đầu tiên:")
print(y.iloc[0])

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


print("\nMa trận X:")
print(X.to_string())
print("\nVector nhãn y:")
print(y.to_string())
X.to_csv(thu_muc / "tables/X.csv", index=False)
y.to_csv(thu_muc / "tables/y.csv", index=False)
data.to_csv(thu_muc / "tables/training_set.csv", index_label="sample_index")


thong_ke = pd.DataFrame()
thong_ke["Min"] = X.min()
thong_ke["Max"] = X.max()
thong_ke["Mean"] = X.mean()
thong_ke["Std"] = X.std(ddof=1)
print("\nBảng thống kê:")
print(thong_ke.round(6))
thong_ke.to_csv(thu_muc / "tables/thong_ke.csv")

# Tính tương quan Pearson cho bốn cột số
tuong_quan = X.corr(method="pearson")
print("\nMa trận Pearson:")
print(tuong_quan.round(6))
tuong_quan.to_csv(thu_muc / "tables/pearson.csv")

# Vẽ heatmap cho Person correlation
plt.figure(figsize=(9, 7))
sns.heatmap(tuong_quan, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Pearson correlation - Iris")
plt.tight_layout()
plt.savefig(thu_muc / "figures/01_heatmap.png", dpi=150)
# Vẽ scatter plot cho 6 cặp đặc trưng
cac_cap = []
for i in range(len(ten_dac_trung)):
    for j in range(i + 1, len(ten_dac_trung)):
        cac_cap.append((ten_dac_trung[i], ten_dac_trung[j]))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

cac_loai = data["species"].unique()
mau_sac = ["tab:blue", "tab:orange", "tab:green"]

for ax, (f1, f2) in zip(axes, cac_cap):
    for loai, mau in zip(cac_loai, mau_sac):
        du_lieu_loai = data[data["species"] == loai]
        ax.scatter(
            du_lieu_loai[f1],
            du_lieu_loai[f2],
            label=loai,
            color=mau,
            alpha=0.7,
            s=30
        )

    r = tuong_quan.loc[f1, f2]
    ax.set_xlabel(f"{f1} (cm)")
    ax.set_ylabel(f"{f2} (cm)")
    ax.set_title(f"{f1} vs {f2}\nr = {r:.2f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Iris: 6 cặp đặc trưng và tương quan Pearson", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(
    thu_muc / "figures/02_pairplot.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()
