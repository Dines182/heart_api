import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg"
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load dataset (yours already does this)
df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Keep feature order for inference later
feature_names = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipelines
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipe_dt = Pipeline([
    ("model", DecisionTreeClassifier(random_state=42))
])

pipe_rf = Pipeline([
    ("model", RandomForestClassifier(random_state=42))
])

# Fit
pipe_lr.fit(X_train, y_train)
pipe_dt.fit(X_train, y_train)
pipe_rf.fit(X_train, y_train)

# Predict
pred_lr = pipe_lr.predict(X_test)
pred_dt = pipe_dt.predict(X_test)
pred_rf = pipe_rf.predict(X_test)

# Evaluate (reuse your helper if you want)
def acc(y_true, y_pred): 
    return accuracy_score(y_true, y_pred)

scores = {
    "Logistic Regression": acc(y_test, pred_lr),
    "Decision Tree":       acc(y_test, pred_dt),
    "Random Forest":       acc(y_test, pred_rf),
}

print("Accuracy scores:", scores)

# Pick best pipeline
best_name = max(scores, key=scores.get)
best_pipe = {"Logistic Regression": pipe_lr, "Decision Tree": pipe_dt, "Random Forest": pipe_rf}[best_name]
print(f"Best model = {best_name}")

# Save the best pipeline (includes preprocessing if any)
joblib.dump(best_pipe, "model.pkl")
print("✅ Saved model.pkl")

# (Optional) save feature names to ensure correct column order at inference
joblib.dump(feature_names, "feature_names.pkl")
print("✅ Saved feature_names.pkl")

# ROC curve for LR (uses pipeline so scaling is handled internally)
try:
    y_probs = pipe_lr.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    # If you're on a headless machine, avoid TkAgg/Qt5Agg; use non-interactive backend or savefig
    plt.figure()
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()   # or plt.savefig("roc.png")
except Exception as e:
    print("Note: ROC plot skipped (predict_proba not available or backend issue).", e)

# Results summary table
results = pd.DataFrame({
    "Model": list(scores.keys()),
    "Accuracy": list(scores.values())
}).sort_values("Accuracy", ascending=False)
print(results)
