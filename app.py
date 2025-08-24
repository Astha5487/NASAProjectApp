import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
import NASAProject1 as np1


# -------------------- Page setup --------------------
st.set_page_config(page_title="NASAProject", layout="wide")
st.title("🚀 NASAProject: Space Materials Survival")
st.write("Upload your MISSE Excel, map the columns, compute **Material Survival Score (MSS)**, and train ML models.")


# -------------------- File upload --------------------
uploaded = st.file_uploader("📂 Upload your MISSE Excel file", type=["xlsx"])
if not uploaded:
    st.info("Upload an Excel file to continue.")
    st.stop()

# Load Excel
sheets, xls = np1.load_excel(uploaded)
sheet = st.sidebar.selectbox("Choose sheet", sheets, index=0)
df = pd.read_excel(xls, sheet_name=sheet)

st.subheader("📊 Data Preview")
st.write(df.head())
with st.expander("Show all columns"):
    st.code("\n".join(df.columns))


# -------------------- Column mapping --------------------
st.sidebar.markdown("### Column Mapping")

def guess_col(cols, patterns):
    for c in cols:
        if any(p in c.lower() for p in patterns):
            return c
    return None

cols = list(df.columns)

label_col = st.sidebar.selectbox("Material Label", ["<none>"] + cols, index=0)
ey_col = st.sidebar.selectbox("Erosion Yield (Ey)", ["<none>"] + cols, index=0)
ml_col = st.sidebar.selectbox("Mass Loss (g)", ["<none>"] + cols, index=0)
th_col = st.sidebar.selectbox("Thickness (mils)", ["<none>"] + cols, index=0)
esh_col = st.sidebar.selectbox("Solar Exposure (ESH)", ["<none>"] + cols, index=0)
ao_col = st.sidebar.selectbox("AO Fluence", ["<none>"] + cols, index=0)

alpha = st.sidebar.slider("α (weight: Ey)", 0.0, 5.0, 1.0, 0.1)
beta = st.sidebar.slider("β (weight: Mass loss / Thickness)", 0.0, 5.0, 1.0, 0.1)
gamma = st.sidebar.slider("γ (weight: Solar / AO)", 0.0, 5.0, 1.0, 0.1)
normalize = st.sidebar.checkbox("Normalize inputs (0-1)", value=True)


# -------------------- Compute MSS --------------------
needed = {"Ey": ey_col, "Mass loss": ml_col, "Thickness": th_col, "Solar": esh_col, "AO": ao_col}
missing = [k for k, v in needed.items() if v == "<none>"]
if missing:
    st.error(f"Please select: {', '.join(missing)}")
    st.stop()

work = df.copy()
for c in [ey_col, ml_col, th_col, esh_col, ao_col]:
    work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0)

if normalize:
    scaler = MinMaxScaler()
    vals = scaler.fit_transform(work[[ey_col, ml_col, th_col, esh_col, ao_col]])
    ey_s, ml_s, th_s, esh_s, ao_s = [vals[:, i] for i in range(5)]
else:
    ey_s = work[ey_col].to_numpy()
    ml_s = work[ml_col].to_numpy()
    th_s = work[th_col].replace(0, np.nan).to_numpy()
    esh_s = work[esh_col].to_numpy()
    ao_s = work[ao_col].replace(0, np.nan).to_numpy()

eps = 1e-9
mss = 1.0 / (1 + alpha*ey_s + beta*(ml_s/(th_s+eps)) + gamma*(esh_s/(ao_s+eps)))
work["MSS"] = np.clip(mss, 0, 1)


# -------------------- Show results --------------------
st.subheader("✅ MSS Computed")
show_cols = ["MSS"]
if label_col != "<none>": show_cols.insert(0, label_col)
st.dataframe(work[show_cols].sort_values("MSS", ascending=False), use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(work[show_cols[0]] if label_col != "<none>" else range(len(work)), work["MSS"])
ax.set_ylabel("MSS (0–1)")
ax.set_title("Ranked Material Survival")
if label_col != "<none>": ax.set_xticklabels(work[label_col].astype(str), rotation=45, ha="right")
st.pyplot(fig)

st.download_button("💾 Download CSV", work.to_csv(index=False).encode("utf-8"), "NASAProject_MSS.csv", "text/csv")


# -------------------- ML Training --------------------
st.subheader("🤖 Train ML Model")
target_cols = st.multiselect("Select target columns for prediction", df.columns)

if st.button("Train Model"):
    if len(target_cols) == 0:
        st.error("⚠ Please select target columns.")
    else:
        regressor, scaler, X_test, y_test = np1.train_models(df, target_cols)
        y_pred = regressor.predict(X_test)

        for i, col in enumerate(target_cols):
            fig = np1.plot_predictions(y_test[col], y_pred[:, i], col)
            st.pyplot(fig)


# -------------------- Extra Visualizations --------------------
st.subheader("📉 Extra Visualizations")
if ao_col != "<none>" and ey_col != "<none>": st.pyplot(np1.plot_ao_vs_ey(work, ao_col, ey_col))
if ao_col != "<none>" and ml_col != "<none>": st.pyplot(np1.plot_ao_vs_massloss(work, ao_col, ml_col))
if th_col != "<none>": st.pyplot(np1.plot_thickness_vs_mss(work, th_col, "MSS"))
