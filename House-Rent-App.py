"""
House Rent Dataset - Linear & Logistic Regression

"""





CSV_PATH = "House_Rent_Dataset.csv"   # change path if needed

# --------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
print(df.head())

# --------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# --------------------------------------------------------------------
# Extract floor number and total floors from "Floor" column, e.g. "1 out of 3"
def parse_floor(f):
    f = str(f)
    if "Ground" in f:
        floor_num = 0
    elif "Basement" in f or "Lower" in f:
        floor_num = -1
    else:
        try:
            floor_num = int(f.split(" out of")[0].strip())
        except ValueError:
            floor_num = np.nan
    try:
        total_floors = int(f.split("out of")[-1].strip())
    except ValueError:
        total_floors = np.nan
    return pd.Series([floor_num, total_floors])

df[["Floor_Num", "Total_Floors"]] = df["Floor"].apply(parse_floor)
df["Floor_Num"] = df["Floor_Num"].fillna(df["Floor_Num"].median())
df["Total_Floors"] = df["Total_Floors"].fillna(df["Total_Floors"].median())

# One-hot encode categorical columns
cat_cols = ["Area Type", "City", "Furnishing Status", "Tenant Preferred", "Point of Contact"]
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Drop columns not useful as numeric features
df_encoded = df_encoded.drop(columns=["Posted On", "Floor", "Area Locality"])

# --------------------------------------------------------------------
# 3. LINEAR REGRESSION  ->  predict Rent
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("LINEAR REGRESSION: Predicting Rent")
print("=" * 60)

X_lin = df_encoded.drop(columns=["Rent"])
y_lin_raw = df_encoded["Rent"]
y_lin = np.log1p(y_lin_raw)   # log-transform: compresses the long right tail (outlier rents)

X_train, X_test, y_train, y_test = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)

scaler_lin = StandardScaler()
X_train_s = scaler_lin.fit_transform(X_train)
X_test_s = scaler_lin.transform(X_test)

lin_model = LinearRegression()
lin_model.fit(X_train_s, y_train)
y_pred_log = lin_model.predict(X_test_s)

# Convert back to actual rupee scale for interpretable metrics
y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_log)

print(f"R^2 Score (log scale) : {r2_score(y_test, y_pred_log):.4f}")
print(f"MAE       : {mean_absolute_error(y_test_actual, y_pred_actual):,.2f}")
print(f"RMSE      : {np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)):,.2f}")

# Coefficients (feature importance direction)
coef_df = pd.DataFrame({
    "Feature": X_lin.columns,
    "Coefficient": lin_model.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print("\nTop coefficients:\n", coef_df.head(10))

# Plot: actual vs predicted (back on real rupee scale)
plt.figure(figsize=(6, 6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.4)
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
plt.xlabel("Actual Rent")
plt.ylabel("Predicted Rent")
plt.title("Linear Regression (log-transformed target): Actual vs Predicted Rent")
plt.tight_layout()
plt.savefig("linear_regression_actual_vs_predicted.png", dpi=150)
plt.close()

# --------------------------------------------------------------------
# 4. LOGISTIC REGRESSION -> predict High_Rent (above/below median)
# --------------------------------------------------------------------
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION: Predicting High_Rent (above median)")
print("=" * 60)

median_rent = df_encoded["Rent"].median()
df_encoded["High_Rent"] = (df_encoded["Rent"] > median_rent).astype(int)
print(f"Median rent threshold: {median_rent:,.0f}")
print(df_encoded["High_Rent"].value_counts())

X_log = df_encoded.drop(columns=["Rent", "High_Rent"])
y_log = df_encoded["High_Rent"]

X_train, X_test, y_train, y_test = train_test_split(
    X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
)

scaler_log = StandardScaler()
X_train_s = scaler_log.fit_transform(X_train)
X_test_s = scaler_log.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_s, y_train)
y_pred_log = log_model.predict(X_test_s)
y_prob_log = log_model.predict_proba(X_test_s)[:, 1]

print(f"Accuracy  : {accuracy_score(y_test, y_pred_log):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_log):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_log):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred_log):.4f}")
print(f"ROC AUC   : {roc_auc_score(y_test, y_prob_log):.4f}")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred_log)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low Rent", "High Rent"])
disp.plot(cmap="Blues")
plt.title("Logistic Regression: Confusion Matrix")
plt.tight_layout()
plt.savefig("logistic_regression_confusion_matrix.png", dpi=150)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob_log)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob_log):.3f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression: ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("logistic_regression_roc_curve.png", dpi=150)
plt.close()

print("\nSaved plots:")
print(" - linear_regression_actual_vs_predicted.png")
print(" - logistic_regression_confusion_matrix.png")
print(" - logistic_regression_roc_curve.png")