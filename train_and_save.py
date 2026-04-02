# ==========================================
# TRAIN WITH FINGERPRINTS + DESCRIPTORS
# ==========================================

import pandas as pd
import joblib
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

from xgboost import XGBClassifier


# =========================
# FEATURE FUNCTION
# =========================
def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Basic descriptors
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol)
    ]

    # Morgan Fingerprint (2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_array = np.array(fp)

    return np.concatenate([desc, fp_array])


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/tox21.csv")

targets = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
    'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]

df[targets] = df[targets].fillna(0)


# =========================
# FEATURE EXTRACTION
# =========================
X = []
valid_idx = []

for i, smi in enumerate(df["smiles"]):
    feat = get_features(smi)
    if feat is not None:
        X.append(feat)
        valid_idx.append(i)

df = df.iloc[valid_idx].reset_index(drop=True)

X = pd.DataFrame(X)

print("Feature shape:", X.shape)  # should be ~2053


# =========================
# TRAIN MODELS
# =========================
models = {}

for t in targets:
    print(f"Training {t}...")

    model = XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric='logloss'
    )

    model.fit(X, df[t])
    models[t] = model


# =========================
# SAVE
# =========================
joblib.dump(models, "tox_models.pkl")
joblib.dump(X.columns, "feature_cols.pkl")

print("Training complete with fingerprints!")