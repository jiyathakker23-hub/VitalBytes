# ==========================================
# TOX21 MULTI-TARGET TOXICITY PREDICTION
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("data/tox21.csv")

print("Original shape:", df.shape)


# ==============================
# 2. FEATURE ENGINEERING (SMILES → DESCRIPTORS)
# ==============================

def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol)
    ]


features = []
valid_indices = []

for i, smi in enumerate(df["smiles"]):
    feat = get_features(smi)
    if feat is not None:
        features.append(feat)
        valid_indices.append(i)


df = df.iloc[valid_indices].reset_index(drop=True)

X = pd.DataFrame(features, columns=[
    "MolWt", "LogP", "HDonors", "HAcceptors", "TPSA"
])

print("Feature shape:", X.shape)


# ==============================
# 3. MULTI-TARGET SETUP
# ==============================

target_columns = [
    "NR-AR",
    "NR-AhR",
    "NR-ER",
    "SR-MMP",
    "SR-p53"
]

# Keep only valid targets
df_targets = df[target_columns].fillna(0)

print("Targets shape:", df_targets.shape)


# ==============================
# 4. TRAIN MULTIPLE MODELS
# ==============================

models = {}
results = {}

for target in target_columns:
    print(f"\nTraining model for: {target}")

    y = df_targets[target]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # prediction
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    models[target] = model
    results[target] = acc

    print(f"{target} Accuracy: {acc:.4f}")


# ==============================
# 5. OVERALL PERFORMANCE
# ==============================

print("\n===== FINAL RESULTS =====")
for k, v in results.items():
    print(f"{k}: {v:.4f}")


# ==============================
# 6. FEATURE IMPORTANCE (GLOBAL INSIGHT)
# ==============================

# Use one representative model (SR-MMP)
ref_model = models["SR-MMP"]

importances = ref_model.feature_importances_
features_names = X.columns

plt.figure()
plt.bar(features_names, importances)
plt.title("Molecular Feature Importance (Toxicity Drivers)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()