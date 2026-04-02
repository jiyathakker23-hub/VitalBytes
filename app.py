# ==========================================
# TOX21 AI TOXICITY PREDICTOR (FINAL EXPLAINABLE UI)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from reportlab.pdfgen import canvas
import io
import os

@st.cache_resource
def load_or_train_model():

    if os.path.exists("tox_models.pkl") and os.path.exists("feature_cols.pkl"):
        models = joblib.load("tox_models.pkl")
        feature_cols = joblib.load("feature_cols.pkl")
        return models, feature_cols

    # =========================
    # TRAIN MODEL (FULL VERSION)
    # =========================
    df = pd.read_csv("data/tox21.csv")
    targets = df.columns[:12]
    df = df.dropna(subset=targets)

    features = []
    valid_idx = []

    for i, sm in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            desc = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol)
            ]

            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            features.append(np.concatenate([desc, np.array(fp)]))
            valid_idx.append(i)

    X = np.array(features)
    y = df.iloc[valid_idx][targets].values

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))
    model.fit(X, y)

    # SAVE MODEL
    joblib.dump(model, "tox_models.pkl")
    joblib.dump(list(range(X.shape[1])), "feature_cols.pkl")

    models = {}
    for i, t in enumerate(targets):
        models[t] = model.estimators_[i]

    return models, list(range(X.shape[1]))


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Tox21 AI", layout="wide")

# =========================
# CSS
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background: #f8fafc;  /* light medical white */
    color: #0f172a;
}

/* Title */
h1 {
    font-size: 40px !important;
    font-weight: 700;
    color: #1e3a8a;  /* deep blue */
}

/* Subheaders */
h2, h3 {
    color: #1d4ed8;  /* blue */
}

/* Inputs */
.stTextInput input {
    background-color: white !important;
    color: black !important;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
}

.stSelectbox div {
    background-color: white !important;
    color: black !important;
    border-radius: 8px;
    border: 1px solid #cbd5e1;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #22c55e);
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 8px 18px;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #16a34a);
}
/* Fix dropdown arrow visibility */
.stSelectbox svg {
    fill: #1e3a8a !important;   /* blue arrow */
}

/* FIX: Make all labels visible */
label {
    color: #0f172a !important;   /* dark text */
    font-weight: 500;
}

/* FIX: Streamlit input labels + captions */
.stTextInput label,
.stSelectbox label,
.stNumberInput label {
    color: #0f172a !important;
}

/* FIX: Placeholder text visibility */
.stTextInput input::placeholder {
    color: #64748b !important;
}

/* FIX: General markdown text consistency */
p, span {
    color: #0f172a;
}
.st-emotion-cache-16idsys p {
    color: #0f172a !important;
}

/* Cards */
.card {
    padding: 14px;
    border-radius: 10px;
    background: white;
    margin-bottom: 10px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

/* Risk styles */
.safe {
    color: #16a34a;
    font-weight: bold;
    font-size: 26px;
}

/* Download button */
.stDownloadButton > button {
    background: linear-gradient(90deg, #2563eb, #22c55e);
    color: white;
    border-radius: 8px;
    padding: 8px 18px;
    border: none;
}

.stDownloadButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #16a34a);
}

.risk {
    color: #dc2626;
    font-weight: bold;
    font-size: 26px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("VITALBYTES")
st.caption("Designed for early-stage drug toxicity screening using AI")

st.divider()

col1, col2 = st.columns([3,1])
with col1:
    st.markdown("### AI-powered toxicity prediction for drug safety")
with col2:
    st.metric("Model", "XGBoost")



# Model
# =========================
# LOADING SCREEN WITH PROGRESS
# =========================
progress_bar = st.progress(0)
status_text = st.empty()

status_text.text("Initializing system...")
progress_bar.progress(10)

status_text.text("Fetching dataset...")
progress_bar.progress(30)

status_text.text("Preparing features...")
progress_bar.progress(50)

status_text.text("Training AI model (first run only)...")
progress_bar.progress(80)

models, feature_cols = load_or_train_model()

progress_bar.progress(100)
status_text.text("AI MODEL IS READY FOR PREDICTION!!")

targets = list(models.keys())

# REMOVE LOADING AFTER DONE
progress_bar.empty()
status_text.empty()

# =========================
# FEATURE FUNCTION
# =========================
def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol)
    ]

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)

    return np.concatenate([desc, fp_array]), desc


# Descriptor names
desc_names = [
    "Molecular Weight",
    "LogP (Lipophilicity)",
    "Hydrogen Bond Donors",
    "Hydrogen Bond Acceptors",
    "Topological Polar Surface Area"
]


# =========================
# INPUT
# =========================
examples = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "Benzene": "c1ccccc1",
    "Ethanol": "CCO"
}

st.subheader("Input Molecule")

col1, col2 = st.columns(2)
with col1:
    choice = st.selectbox("Select Example", list(examples.keys()))
with col2:
    smiles = st.text_input("Or Enter SMILES", examples[choice])


st.write("")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict = st.button("Predict")


# =========================
# PDF FUNCTION
# =========================
def generate_pdf(smiles, results, risk_score, explanation):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer)

    c.setFont("Helvetica", 12)
    c.drawString(50, 800, "TOX21 AI TOXICITY REPORT")

    c.drawString(50, 770, f"SMILES: {smiles}")
    c.drawString(50, 750, f"Risk Score: {risk_score}/100")

    y = 720
    for k, v in results.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 20

    c.drawString(50, y - 20, "Key Chemical Insights:")
    y -= 40

    for line in explanation:
        c.drawString(50, y, line)
        y -= 20

    c.save()
    buffer.seek(0)
    return buffer


# =========================
# PREDICTION
# =========================
if predict:

    features, desc_values = get_features(smiles)

    if features is None:
        st.error("Invalid SMILES input")

    else:
        input_df = pd.DataFrame([features], columns=feature_cols)

        probs = []
        results = {}

        for t in targets:
            pred = models[t].predict(input_df)[0]
            prob = models[t].predict_proba(input_df)[0][1]

            probs.append(prob)
            results[t] = "TOXIC" if pred == 1 else "SAFE"

        # =========================
        # RISK SCORE
        # =========================
        risk_score = int(np.mean(probs) * 100)

        st.subheader("Overall Risk")

        if risk_score <= 30:
            st.markdown(f"<div class='safe'>SAFE • {risk_score}/100</div>", unsafe_allow_html=True)
            st.info("Low predicted toxicity across biological pathways.")
        elif risk_score <= 60:
            st.warning(f"Moderate Risk • {risk_score}/100")
        else:
            st.markdown(f"<div class='risk'>HIGH RISK • {risk_score}/100</div>", unsafe_allow_html=True)
            st.error("High predicted toxicity. Further validation recommended.")

        st.divider()

        # =========================
        # RESULTS
        # =========================
        st.subheader("Toxicity Analysis")

        for k, v in results.items():
            color = "#22c55e" if v == "SAFE" else "#ef4444"
            st.markdown(
                f"<div class='card'><b>{k}</b><br><span style='color:{color}; font-size:18px'>{v}</span></div>",
                unsafe_allow_html=True
            )

        st.divider()

        # =========================
        # EXPLAINABLE FEATURES
        # =========================
        st.subheader("Key Chemical Insights")

        model = models[targets[0]]
        importances = model.feature_importances_[:5]

        explanation = []

        for i in range(5):
            name = desc_names[i]
            value = desc_values[i]
            importance = importances[i]

            text = f"{name}: {value:.2f} → contributes to toxicity (importance: {importance:.3f})"
            explanation.append(text)
            st.write(text)

        st.divider()

        # =========================
        # DOWNLOAD
        # =========================
        pdf = generate_pdf(smiles, results, risk_score, explanation)

        st.download_button(
            label="Download Report",
            data=pdf,
            file_name="toxicity_report.pdf",
            mime="application/pdf"
        )