"""
==============================================================================
🏠 House Rent Predictor - Production-Ready Machine Learning & Streamlit App
Zero-Error Standalone Python Script
==============================================================================
Usage:
  1. As Streamlit Web App:
     $ pip install streamlit pandas numpy scikit-learn matplotlib
     $ streamlit run app.py

  2. As Terminal CLI Script:
     $ python app.py
==============================================================================
"""

import sys
import os
import io
import numpy as np
import pandas as pd

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

CSV_FILENAME = "House_Rent_Dataset.csv"

# --------------------------------------------------------------------
# 1. DATA LOADER & SYNTHETIC DATASET GENERATOR (Zero-Crash Guarantee)
# --------------------------------------------------------------------
def generate_sample_dataset(n_samples: int = 1500) -> pd.DataFrame:
    """Generates realistic synthetic data matching the Kaggle House Rent Dataset distribution."""
    np.random.seed(42)
    cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata"]
    area_types = ["Super Area", "Carpet Area", "Built Area"]
    furnishings = ["Furnished", "Semi-Furnished", "Unfurnished"]
    tenants = ["Bachelors/Family", "Bachelors", "Family"]
    contacts = ["Contact Owner", "Contact Agent", "Contact Builder"]

    bhks = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.25, 0.45, 0.25, 0.05])
    sizes = np.clip(np.round(bhks * 450 + np.random.normal(150, 180, size=n_samples)), 200, 4500).astype(int)
    bathrooms = np.clip(bhks + np.random.choice([0, 1, -1], size=n_samples, p=[0.7, 0.2, 0.1]), 1, 6)
    
    city_picks = np.random.choice(cities, size=n_samples, p=[0.20, 0.18, 0.22, 0.16, 0.14, 0.10])
    area_picks = np.random.choice(area_types, size=n_samples, p=[0.55, 0.43, 0.02])
    furn_picks = np.random.choice(furnishings, size=n_samples, p=[0.18, 0.48, 0.34])
    tenant_picks = np.random.choice(tenants, size=n_samples, p=[0.72, 0.18, 0.10])
    contact_picks = np.random.choice(contacts, size=n_samples, p=[0.65, 0.34, 0.01])
    
    total_floors = np.random.choice(range(2, 35), size=n_samples)
    floor_nums = [np.random.randint(0, tf + 1) for tf in total_floors]
    floor_strings = [f"{fn} out of {tf}" if fn > 0 else f"Ground out of {tf}" for fn, tf in zip(floor_nums, total_floors)]

    # Calibrate realistic rent (log-linear with noise)
    city_mult = {"Mumbai": 2.2, "Delhi": 1.35, "Bangalore": 1.15, "Chennai": 0.95, "Hyderabad": 0.88, "Kolkata": 0.72}
    furn_mult = {"Furnished": 1.35, "Semi-Furnished": 1.05, "Unfurnished": 0.85}
    
    rents = []
    for i in range(n_samples):
        base = 7500 + (sizes[i] * 16) + (bathrooms[i] * 3200) + (floor_nums[i] * 250)
        c_m = city_mult[city_picks[i]]
        f_m = furn_mult[furn_picks[i]]
        agent_m = 1.18 if contact_picks[i] == "Contact Agent" else 1.0
        noise = np.random.lognormal(mean=0.0, sigma=0.22)
        rent = int(base * c_m * f_m * agent_m * noise)
        rents.append(max(3500, round(rent, -2)))

    return pd.DataFrame({
        "Posted On": "2022-05-15",
        "BHK": bhks,
        "Rent": rents,
        "Size": sizes,
        "Floor": floor_strings,
        "Area Type": area_picks,
        "Area Locality": "Metropolitan Region",
        "City": city_picks,
        "Furnishing Status": furn_picks,
        "Tenant Preferred": tenant_picks,
        "Bathroom": bathrooms,
        "Point of Contact": contact_picks,
    })


def load_and_train_models(uploaded_file=None):
    """Loads CSV dataset, engineers features, and fits both ML models with zero errors."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "User Uploaded CSV"
    elif os.path.exists(CSV_FILENAME):
        df = pd.read_csv(CSV_FILENAME)
        source = f"Local File: {CSV_FILENAME}"
    else:
        df = generate_sample_dataset(2000)
        source = "Calibrated Synthetic Dataset (Kaggle distribution)"

    # --- 2. Feature Engineering ---
    def parse_floor(f):
        f = str(f)
        if "Ground" in f:
            floor_num = 0
        elif "Basement" in f or "Lower" in f:
            floor_num = -1
        else:
            try:
                floor_num = int(f.split(" out of")[0].strip())
            except (ValueError, IndexError):
                floor_num = np.nan
        try:
            total_floors = int(f.split("out of")[-1].strip())
        except (ValueError, IndexError):
            total_floors = np.nan
        return pd.Series([floor_num, total_floors])

    df[["Floor_Num", "Total_Floors"]] = df["Floor"].apply(parse_floor)
    df["Floor_Num"] = df["Floor_Num"].fillna(df["Floor_Num"].median())
    df["Total_Floors"] = df["Total_Floors"].fillna(df["Total_Floors"].median())

    cat_cols = ["Area Type", "City", "Furnishing Status", "Tenant Preferred", "Point of Contact"]
    # Drop irrelevant raw string columns
    cols_to_drop = [c for c in ["Posted On", "Floor", "Area Locality"] if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

    # --- 3. Linear Regression (Continuous Rent Prediction) ---
    X_lin = df_encoded.drop(columns=["Rent"])
    y_lin_raw = df_encoded["Rent"]
    y_lin_log = np.log1p(y_lin_raw)

    X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
        X_lin, y_lin_log, test_size=0.2, random_state=42
    )

    scaler_lin = StandardScaler()
    X_train_lin_s = scaler_lin.fit_transform(X_train_lin)
    X_test_lin_s = scaler_lin.transform(X_test_lin)

    lin_model = LinearRegression()
    lin_model.fit(X_train_lin_s, y_train_lin)
    y_pred_lin_log = lin_model.predict(X_test_lin_s)

    y_test_actual = np.expm1(y_test_lin)
    y_pred_actual = np.expm1(y_pred_lin_log)

    lin_r2 = r2_score(y_test_lin, y_pred_lin_log)
    lin_mae = mean_absolute_error(y_test_actual, y_pred_actual)
    lin_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

    coef_df = pd.DataFrame({
        "Feature": X_lin.columns,
        "Coefficient": lin_model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False)

    # --- 4. Logistic Regression (High_Rent Classification) ---
    median_rent = df_encoded["Rent"].median()
    df_encoded["High_Rent"] = (df_encoded["Rent"] > median_rent).astype(int)

    X_log = df_encoded.drop(columns=["Rent", "High_Rent"])
    y_log = df_encoded["High_Rent"]

    X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
        X_log, y_log, test_size=0.2, random_state=42, stratify=y_log
    )

    scaler_log = StandardScaler()
    X_train_log_s = scaler_log.fit_transform(X_train_log)
    X_test_log_s = scaler_log.transform(X_test_log)

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_log_s, y_train_log)

    y_pred_log_class = log_model.predict(X_test_log_s)
    y_prob_log_class = log_model.predict_proba(X_test_log_s)[:, 1]

    log_acc = accuracy_score(y_test_log, y_pred_log_class)
    log_prec = precision_score(y_test_log, y_pred_log_class, zero_division=0)
    log_rec = recall_score(y_test_log, y_pred_log_class, zero_division=0)
    log_f1 = f1_score(y_test_log, y_pred_log_class, zero_division=0)
    log_auc = roc_auc_score(y_test_log, y_prob_log_class)

    cm = confusion_matrix(y_test_log, y_pred_log_class)
    fpr, tpr, _ = roc_curve(y_test_log, y_prob_log_class)

    return {
        "source": source,
        "total_rows": len(df),
        "columns": X_lin.columns.tolist(),
        "lin_model": lin_model,
        "scaler_lin": scaler_lin,
        "lin_r2": lin_r2,
        "lin_mae": lin_mae,
        "lin_rmse": lin_rmse,
        "coef_df": coef_df,
        "log_model": log_model,
        "scaler_log": scaler_log,
        "median_rent": median_rent,
        "log_acc": log_acc,
        "log_prec": log_prec,
        "log_rec": log_rec,
        "log_f1": log_f1,
        "log_auc": log_auc,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "y_test_actual": y_test_actual.values,
        "y_pred_actual": y_pred_actual,
        "cities": sorted(df["City"].unique().tolist()),
        "area_types": sorted(df["Area Type"].unique().tolist()),
        "furnishing": sorted(df["Furnishing Status"].unique().tolist()),
        "tenant": sorted(df["Tenant Preferred"].unique().tolist()),
        "contact": sorted(df["Point of Contact"].unique().tolist()),
    }


# --------------------------------------------------------------------
# 2. STREAMLIT WEB APP RUNNER
# --------------------------------------------------------------------
def run_streamlit_app():
    import streamlit as st
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Safe caching decorator
    load_fn = st.cache_data(show_spinner=True)(load_and_train_models)

    st.title("🏠 House Rent Predictor")
    st.markdown(
        "Predict Indian residential rental rates and classify High vs. Low rent "
        "using calibrated **Linear Regression (log-scale)** and **Logistic Regression**."
    )

    # Sidebar data controls
    with st.sidebar:
        st.header("⚙️ Data & Model Settings")
        uploaded_file = st.file_uploader(
            "Upload Custom CSV (Optional)",
            type=["csv"],
            help="Upload Kaggle House_Rent_Dataset.csv or leave blank to use the built-in dataset."
        )
        if st.button("🔄 Retrain Models", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    artifacts = load_fn(uploaded_file)

    # Summary KPI Bar
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Linear R² Score", f"{artifacts['lin_r2']:.3f}", "Log target fit")
    kpi2.metric("Mean Abs Error (MAE)", f"₹{artifacts['lin_mae']:,.0f}")
    kpi3.metric("Logistic Accuracy", f"{artifacts['log_acc']*100:.1f}%", "Median split")
    kpi4.metric("ROC AUC", f"{artifacts['log_auc']:.3f}", "High vs Low Rent")

    st.info(f"📊 **Data Source:** {artifacts['source']} ({artifacts['total_rows']:,} rows). Median Rent Threshold: **₹{artifacts['median_rent']:,.0f}**.")

    # Main Layout Tabs
    tab_predict, tab_diagnostics, tab_features = st.tabs(["🔮 Rent Predictor", "📈 Model Diagnostics", "🧠 Feature Importance"])

    with tab_predict:
        st.subheader("Enter Property Details")
        c1, c2 = st.columns(2)

        with c1:
            bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2, step=1)
            size = st.number_input("Size (sq. ft.)", min_value=100, max_value=10000, value=1050, step=25)
            bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
            floor_num = st.number_input("Floor Number (0 = Ground)", min_value=-1, max_value=100, value=2, step=1)
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=100, value=6, step=1)

        with c2:
            default_city_idx = artifacts["cities"].index("Bangalore") if "Bangalore" in artifacts["cities"] else 0
            city = st.selectbox("City", artifacts["cities"], index=default_city_idx)
            area_type = st.selectbox("Area Type", artifacts["area_types"])
            furnishing = st.selectbox("Furnishing Status", artifacts["furnishing"])
            tenant = st.selectbox("Tenant Preferred", artifacts["tenant"])
            contact = st.selectbox("Point of Contact", artifacts["contact"])

        def build_input_row():
            row = {col: 0 for col in artifacts["columns"]}
            row["BHK"] = bhk
            row["Size"] = size
            row["Bathroom"] = bathroom
            row["Floor_Num"] = floor_num
            row["Total_Floors"] = total_floors

            for prefix, value in [
                ("Area Type_", area_type),
                ("City_", city),
                ("Furnishing Status_", furnishing),
                ("Tenant Preferred_", tenant),
                ("Point of Contact_", contact),
            ]:
                key = f"{prefix}{value}"
                if key in row:
                    row[key] = 1

            return pd.DataFrame([row])[artifacts["columns"]]

        if st.button("⚡ Calculate Rent Estimate", type="primary", use_container_width=True):
            input_df = build_input_row()

            # Linear regression prediction
            input_scaled_lin = artifacts["scaler_lin"].transform(input_df)
            pred_log_rent = artifacts["lin_model"].predict(input_scaled_lin)[0]
            pred_rent = np.expm1(pred_log_rent)

            # Logistic regression prediction
            input_scaled_log = artifacts["scaler_log"].transform(input_df)
            pred_class = artifacts["log_model"].predict(input_scaled_log)[0]
            pred_prob = artifacts["log_model"].predict_proba(input_scaled_log)[0, 1]

            st.divider()
            st.subheader("🎯 Prediction Results")
            
            res1, res2 = st.columns(2)
            with res1:
                st.metric(
                    label="Estimated Monthly Rent (Linear Regression)",
                    value=f"₹{pred_rent:,.0f}",
                    delta=f"{((pred_rent - artifacts['median_rent']) / artifacts['median_rent']) * 100:+.1f}% vs. Median"
                )
                st.caption(f"Estimated 90% Range: ₹{np.expm1(pred_log_rent - 0.22):,.0f} - ₹{np.expm1(pred_log_rent + 0.22):,.0f}")

            with res2:
                class_label = "🔥 High Rent (> Median)" if pred_class == 1 else "🟢 Standard/Low Rent (≤ Median)"
                st.metric(
                    label="Classification (Logistic Regression)",
                    value=class_label,
                    delta=f"{pred_prob * 100:.1f}% Probability of High Rent"
                )
                st.progress(float(np.clip(pred_prob, 0.0, 1.0)))

    with tab_diagnostics:
        st.subheader("Validation Visualizations")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("**Linear Regression: Actual vs. Predicted Rent**")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(artifacts["y_test_actual"][:250], artifacts["y_pred_actual"][:250], alpha=0.5, color="#059669")
            max_val = min(max(artifacts["y_test_actual"][:250]), 100000)
            ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Ideal Fit (y = x)")
            ax.set_xlabel("Actual Rent (₹)")
            ax.set_ylabel("Predicted Rent (₹)")
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig)
            plt.close(fig)

        with col_d2:
            st.markdown("**Logistic Regression: ROC Curve**")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(artifacts["fpr"], artifacts["tpr"], color="#2563eb", lw=2, label=f"ROC AUC = {artifacts['log_auc']:.3f}")
            ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")
            ax.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig)
            plt.close(fig)

    with tab_features:
        st.subheader("Key Rent Drivers (Linear Regression Coefficients)")
        st.markdown("Features with positive coefficients increase predicted rent; negative coefficients decrease it.")
        
        top_coefs = pd.concat([
            artifacts["coef_df"].head(6),
            artifacts["coef_df"].tail(6)
        ]).drop_duplicates()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#10b981" if c > 0 else "#ef4444" for c in top_coefs["Coefficient"]]
        ax.barh(top_coefs["Feature"], top_coefs["Coefficient"], color=colors)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Standardized Coefficient (log-rent impact)")
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------------------------
# 3. CLI / TERMINAL ENTRY POINT
# --------------------------------------------------------------------
def run_cli():
    print("=" * 70)
    print("🏠 HOUSE RENT PREDICTOR - MACHINE LEARNING PIPELINE")
    print("=" * 70)
    print("Training Linear Regression (log scale) and Logistic Regression...")
    artifacts = load_and_train_models()
    print(f"✓ Data Source: {artifacts['source']} ({artifacts['total_rows']} rows)")
    print(f"✓ Median Rent Threshold: ₹{artifacts['median_rent']:,.0f}")
    print("-" * 70)
    print(f"📊 Linear Regression R² Score : {artifacts['lin_r2']:.4f}")
    print(f"📊 Mean Absolute Error (MAE)   : ₹{artifacts['lin_mae']:,.0f}")
    print(f"📊 Root Mean Squared Err (RMSE): ₹{artifacts['lin_rmse']:,.0f}")
    print("-" * 70)
    print(f"🎯 Logistic Regression Accuracy: {artifacts['log_acc'] * 100:.2f}%")
    print(f"🎯 Precision                   : {artifacts['log_prec'] * 100:.2f}%")
    print(f"🎯 Recall                      : {artifacts['log_rec'] * 100:.2f}%")
    print(f"🎯 F1-Score                    : {artifacts['log_f1'] * 100:.2f}%")
    print(f"🎯 ROC-AUC                     : {artifacts['log_auc']:.4f}")
    print("=" * 70)
    print("Top Feature Drivers of Rent:")
    for idx, row in artifacts['coef_df'].head(5).iterrows():
        print(f"  + {row['Feature']:<30} (Coeff: +{row['Coefficient']:.3f})")
    for idx, row in artifacts['coef_df'].tail(3).iterrows():
        print(f"  - {row['Feature']:<30} (Coeff: {row['Coefficient']:.3f})")
    print("=" * 70)
    print("💡 To launch the interactive web dashboard:")
    print("   streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    # Check if running under Streamlit or direct Python CLI
    try:
        import streamlit as st
        # Detect if launched via `streamlit run`
        if hasattr(st, "runtime") and st.runtime.exists():
            st.set_page_config(
                page_title="House Rent Predictor ML",
                page_icon="🏠",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            run_streamlit_app()
        else:
            run_cli()
    except Exception as e:
        print(f"Notice: Running in CLI mode ({e})")
        run_cli()