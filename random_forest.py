import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("Hackathon data - Sheet1.csv")

df["target"] = df["TRX_TYPE_AFTER_HP"].apply(
    lambda x: 0 if x == "visoka trx aktivnost" else 1
)

X = df.drop(columns=["CLIENT_ID", "TRX_TYPE_AFTER_HP", "target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
)

model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

for thr in [0.50, 0.40, 0.30, 0.25]:
    y_pred_thr = (y_proba >= thr).astype(int)
    print(f"\n=== Threshold {thr} ===")
    print(classification_report(y_test, y_pred_thr, digits=3))


X_test_sample = X_test.sample(4000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values[1], X_test_sample)
