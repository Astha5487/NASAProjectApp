import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib


# -------------------- Load Excel --------------------
def load_excel(path):
    xls = pd.ExcelFile(path)
    return xls.sheet_names, xls


# -------------------- Train models --------------------
def train_models(df, target_cols):
    """
    Train regression model(s) on selected target columns.
    Saves scaler + model with joblib.
    """
    # Features = numeric columns except target(s)
    X = df.drop(columns=target_cols, errors="ignore").select_dtypes(include="number")
    y = df[target_cols].copy()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Regression model
    regressor = LinearRegression()
    regressor.fit(X_train_scaled, y_train)

    # Save models
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(regressor, "reg_model.pkl")

    return regressor, scaler, X_test_scaled, y_test


# -------------------- Plot helper --------------------
def plot_predictions(y_true, y_pred, target_name):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.7)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"Predictions vs True ({target_name})")
    return fig


# -------------------- Extra Visualization Helpers --------------------
def plot_ao_vs_ey(df, ao_col, ey_col):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[ao_col], df[ey_col], alpha=0.6, c="blue")
    ax.set_xlabel("AO Fluence (atoms/cm²)")
    ax.set_ylabel("Erosion Yield (cm³/atom)")
    ax.set_title("AO Fluence vs Erosion Yield")
    return fig


def plot_ao_vs_massloss(df, ao_col, ml_col):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[ao_col], df[ml_col], alpha=0.6, c="green")
    ax.set_xlabel("AO Fluence (atoms/cm²)")
    ax.set_ylabel("Mass Loss (g)")
    ax.set_title("AO Fluence vs Mass Loss")
    return fig


def plot_thickness_vs_mss(df, th_col, mss_col):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[th_col], df[mss_col], alpha=0.6, c="red")
    ax.set_xlabel("Thickness (mils)")
    ax.set_ylabel("MSS (0–1)")
    ax.set_title("Thickness vs MSS")
    return fig
