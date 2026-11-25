from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="FraudShield — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1.5rem !important;
    }
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
        padding: 1rem 1.3rem;
        border-radius: 10px;
        color: #ffffff;
        margin-bottom: 0.7rem;
    }
    .hero-box h1 {
        color: #ffffff !important;
        font-size: 1.45rem !important;
        margin: 0 0 0.25rem 0 !important;
        line-height: 1.3 !important;
    }
    .hero-box p {
        color: #dbeafe;
        margin: 0;
        font-size: 0.9rem;
    }
    .result-card-fraud {
        background: #fef2f2;
        border: 1.5px solid #ef4444;
        border-left: 5px solid #dc2626;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        color: #7f1d1d;
        margin-bottom: 0.4rem;
    }
    .result-card-normal {
        background: #f0fdf4;
        border: 1.5px solid #22c55e;
        border-left: 5px solid #16a34a;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        color: #14532d;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path("app/best_model.pkl")
DATA_PATH = Path("data/dataset.csv")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_sample_rows(feature_cols):
    normal_sample = {col: 0.0 for col in feature_cols}
    fraud_sample = {col: 0.0 for col in feature_cols}
    if not DATA_PATH.exists():
        return normal_sample, fraud_sample

    df_sub = pd.read_csv(DATA_PATH, nrows=50000)
    target_col = "Class" if "Class" in df_sub.columns else df_sub.columns[-1]
    if target_col in df_sub.columns:
        norm_rows = df_sub[df_sub[target_col] == 0]
        fraud_rows = df_sub[df_sub[target_col] == 1]
        if not norm_rows.empty:
            normal_sample = {col: float(norm_rows.iloc[0].get(col, 0.0)) for col in feature_cols}
        if not fraud_rows.empty:
            fraud_sample = {col: float(fraud_rows.iloc[0].get(col, 0.0)) for col in feature_cols}
    return normal_sample, fraud_sample


if not MODEL_PATH.exists():
    st.error("File `app/best_model.pkl` belum ditemukan. Jalankan `python train.py` terlebih dahulu.")
    st.stop()

artifact = load_artifact()
model = artifact["model"]
features = artifact["features"]
best_name = artifact.get("best_name", "RandomForest")

normal_sample, fraud_sample = load_sample_rows(features)

for feat in features:
    st.session_state.setdefault(f"feat_{feat}", float(normal_sample.get(feat, 0.0)))

with st.sidebar:
    st.markdown("## 🛡️ FraudShield ML")
    st.caption("Deteksi Penipuan Transaksi Kartu Kredit")
    st.divider()
    st.markdown("### Ringkasan Model")
    st.success(f"Model Aktif: **{best_name}**")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("PR-AUC", "0.8511")
    col_m2.metric("Jumlah Fitur", f"{len(features)}")
    st.divider()
    st.markdown("### Info Dataset & Target")
    st.write("- **Dataset:** Credit Card Fraud (`dataset.csv`)")
    st.write("- **Kelas 0:** Transaksi Normal")
    st.write("- **Kelas 1:** Terindikasi Fraud")

st.markdown(
    f"""
    <div class="hero-box">
        <h1>🛡️ FraudShield — Deteksi Penipuan Kartu Kredit</h1>
        <p>Evaluasi transaksi tunggal maupun file CSV massal menggunakan model <strong>{best_name}</strong> ({len(features)} fitur transaksi).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["⚡ Prediksi Satu Transaksi", "📂 Prediksi Batch (Upload CSV)"])

with tab1:
    st.subheader("Uji Cepat dengan Contoh Data")
    b1, b2, b3 = st.columns([1, 1, 2])
    if b1.button("Isi Contoh Normal", use_container_width=True):
        for feat in features:
            st.session_state[f"feat_{feat}"] = float(normal_sample.get(feat, 0.0))
        st.rerun()

    if b2.button("Isi Contoh Fraud", use_container_width=True):
        for feat in features:
            st.session_state[f"feat_{feat}"] = float(fraud_sample.get(feat, 0.0))
        st.rerun()

    with st.expander(f"Lihat / Ubah {len(features)} Nilai Fitur Transaksi", expanded=False):
        cols = st.columns(4)
        for idx, feat in enumerate(features):
            with cols[idx % 4]:
                st.number_input(
                    feat,
                    step=0.1,
                    format="%.4f",
                    key=f"feat_{feat}",
                )

    if st.button("🔍 Analisis Transaksi Sekarang", type="primary"):
        row_data = {feat: float(st.session_state[f"feat_{feat}"]) for feat in features}
        X_input = pd.DataFrame([row_data])[features]
        pred = int(model.predict(X_input)[0])
        proba = float(model.predict_proba(X_input)[0][1]) if hasattr(model, "predict_proba") else float(pred)
        risk_score = int(round(proba * 100))

        st.markdown("### Hasil Analisis")
        if pred == 1:
            st.markdown(
                f"""
                <div class="result-card-fraud">
                    <strong>⚠️ RISIKO TINGGI — Terindikasi Fraud (Kelas 1)</strong><br>
                    Probabilitas Fraud: <strong>{proba:.2%}</strong> (Skor Risiko: <strong>{risk_score}/100</strong>)
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-card-normal">
                    <strong>✅ RISIKO RENDAH — Transaksi Normal (Kelas 0)</strong><br>
                    Probabilitas Fraud: <strong>{proba:.2%}</strong> (Skor Risiko: <strong>{risk_score}/100</strong>)
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.progress(min(max(proba, 0.0), 1.0))

with tab2:
    st.subheader("Prediksi Massal dari File CSV")
    uploaded = st.file_uploader("Pilih file CSV transaksi", type=["csv"])
    max_rows = st.slider("Jumlah baris yang diproses", min_value=10, max_value=5000, value=200, step=10)

    if uploaded is not None:
        df_up = pd.read_csv(uploaded).head(max_rows)
        missing = [c for c in features if c not in df_up.columns]
        if missing:
            st.warning(f"Kolom berikut tidak ada di CSV dan diisi 0.0: {missing}")
            for c in missing:
                df_up[c] = 0.0

        X_batch = df_up[features]
        preds = model.predict(X_batch)
        probs = model.predict_proba(X_batch)[:, 1] if hasattr(model, "predict_proba") else preds.astype(float)

        fraud_count = int((preds == 1).sum())
        normal_count = int((preds == 0).sum())

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Baris Diuji", len(df_up))
        k2.metric("Transaksi Normal", normal_count)
        k3.metric("Terindikasi Fraud", fraud_count)

        result_df = df_up.copy()
        result_df["Probabilitas_Fraud"] = np.round(probs, 4)
        result_df["Status_Prediksi"] = np.where(preds == 1, "Fraud (1)", "Normal (0)")

        st.dataframe(result_df.head(50), use_container_width=True)

        csv_out = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Hasil Prediksi (CSV)",
            data=csv_out,
            file_name="hasil_prediksi_fraud.csv",
            mime="text/csv",
        )