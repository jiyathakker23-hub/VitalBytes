#  VitalBytes - AI Drug Toxicity Predictor

VitalBytes is an AI-powered web application that predicts chemical toxicity using molecular structure (SMILES format). It helps in early-stage drug safety screening.

---

##  Features
- SMILES-based molecule input
- Multi-target toxicity prediction (Tox21 dataset)
- Risk scoring system (Safe / Moderate / High Risk)
- Explainable AI insights (chemical descriptors)
- PDF report generation

---

##  Example Molecules
- Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`
- Paracetamol: `CC(=O)NC1=CC=C(O)C=C1`
- Benzene: `c1ccccc1`
- Ethanol: `CCO`

---

##  Tech Stack
- Python
- Streamlit
- RDKit
- Scikit-learn
- Pandas, NumPy
- ReportLab

---

##  How to Run Locally

```bash
git clone https://github.com/jiyathakker23-hub/VitalBytes.git
cd VitalBytes
pip install -r requirements.txt
streamlit run app.py