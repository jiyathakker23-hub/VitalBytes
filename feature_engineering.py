import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load data
df = pd.read_csv("data/tox21.csv")

# Choose SMILES column
smiles_col = "smiles"


def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return [
        Descriptors.MolWt(mol),  # molecular weight
        Descriptors.MolLogP(mol),  # lipophilicity
        Descriptors.NumHDonors(mol),  # H bond donors
        Descriptors.NumHAcceptors(mol),  # H bond acceptors
        Descriptors.TPSA(mol)  # polar surface area
    ]


features = []
valid_indices = []

for i, smi in enumerate(df[smiles_col]):
    feat = get_features(smi)
    if feat is not None:
        features.append(feat)
        valid_indices.append(i)

# Create feature dataframe
X = pd.DataFrame(features, columns=[
    "MolWt", "LogP", "HDonors", "HAcceptors", "TPSA"
])

print(X.head())
print("Feature shape:", X.shape)