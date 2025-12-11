import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Učitavanje podataka
df = pd.read_csv("Hackathon data - Sheet1.csv")

print("=== 1. OSNOVNE INFORMACIJE ===")
# Ukupan broj klijenata
ukupan_broj = len(df)
print(f"Ukupan broj klijenata u setu: {ukupan_broj}")

status_counts = df["TRX_TYPE_AFTER_HP"].value_counts()
status_percentages = df["TRX_TYPE_AFTER_HP"].value_counts(normalize=True) * 100

print("\nDistribucija ponašanja klijenata:")
print(status_counts)

# Vizualizacija
plt.figure(figsize=(10, 6))
sns.countplot(
    y="TRX_TYPE_AFTER_HP", data=df, order=status_counts.index, palette="viridis"
)
plt.title("Distribucija statusa klijenata nakon povijesnog perioda")
plt.xlabel("Broj klijenata")
plt.ylabel("Status")
plt.show()

# Koliko ih je "promijenilo" ponašanje?
# Pretpostavljamo da je 'visoka trx aktivnost' nastavak, a sve ostalo promjena
loyal = status_counts["visoka trx aktivnost"]
risk = ukupan_broj - loyal
print(f"\nNastavili 'visoku aktivnost': {loyal} ({loyal / ukupan_broj:.1%})")
print(f"Promijenili ponašanje (Rizik/Churn): {risk} ({risk / ukupan_broj:.1%})")


print("\n=== 2. ANALIZA PROIZVODA ===")
df["target_label"] = df["TRX_TYPE_AFTER_HP"].apply(
    lambda x: "Lojalan (Visoka Akt.)"
    if x == "visoka trx aktivnost"
    else "Rizičan (Smanjena/Prekid)"
)

product_cols = [col for col in df.columns if "_flag" in col]
product_analysis = df.groupby("target_label")[product_cols].mean() * 100

print("Postotak posjedovanja proizvoda po grupama:")
print(product_analysis.T)

plt.figure(figsize=(12, 8))
sns.heatmap(product_analysis.T, annot=True, fmt=".1f", cmap="RdBu_r", center=30)
plt.show()


print("\n=== 3. FINANCIJSKI TRENDOVI (DIFF STUPCI) ===")

# Izdvajamo stupce koji prate promjene (završavaju na _diff)
diff_cols = [col for col in df.columns if "_diff" in col]
trend_analysis = df.groupby("target_label")[diff_cols].median()

print("Medijan promjene (Diff) po grupama:")
print(trend_analysis.T)

plt.figure(figsize=(6, 5))
sns.boxplot(x="target_label", y="AGE_decile", data=df, showfliers=False)
plt.title("Distribucija dobi po grupama")

age_decile_counts = (
    df.groupby(["target_label", "AGE_decile"]).size().unstack(fill_value=0)
)
print("\nBroj klijenata po dobnim decilima i grupama:")
print(age_decile_counts)


cols_to_plot = ["N_TRX_diff", "VOL_TRX_diff", "MB_APP_login_diff"]

plt.figure(figsize=(15, 5))
for i, col in enumerate(cols_to_plot):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x="target_label", y=col, data=df, showfliers=False)
    plt.title(f"Razlika u: {col}")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
