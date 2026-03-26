"""
Step 2: Feature Calculation
Inputs:  data_cleaned.csv
Outputs: features/X_ecfp4.npy
         features/X_desc.npy
         features/X_combined.npy
         features/desc_imputer.pkl
         features/desc_selector.pkl
         features/desc_names.txt
"""

import os
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

os.makedirs("features", exist_ok=True)

df = pd.read_csv("data_cleaned.csv")
print(f"化合物数: {len(df)}")

# ── 2.1 Morgan Fingerprint (ECFP4) ────────────────────────────────────────────
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def mol_to_ecfp4(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(2048, dtype=np.uint8)
    return _morgan_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)

X_ecfp4 = np.vstack(df["smiles_std"].apply(mol_to_ecfp4).values)
np.save("features/X_ecfp4.npy", X_ecfp4)
print(f"ECFP4 特征矩阵: {X_ecfp4.shape}  → features/X_ecfp4.npy")

# ── 2.2 RDKit 2D Descriptors ──────────────────────────────────────────────────
desc_names = [name for name, _ in Descriptors.descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

def mol_to_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * len(desc_names)
    return list(calculator.CalcDescriptors(mol))

X_desc_raw = np.array(
    df["smiles_std"].apply(mol_to_descriptors).tolist(),
    dtype=float
)

# Impute and filter
imputer = SimpleImputer(strategy="median")
X_desc_imputed = imputer.fit_transform(X_desc_raw)

selector = VarianceThreshold(threshold=0.01)
X_desc = selector.fit_transform(X_desc_imputed)

# Keep track of selected descriptor names
selected_mask = selector.get_support()
selected_desc_names = [name for name, keep in zip(desc_names, selected_mask) if keep]

np.save("features/X_desc.npy", X_desc)
joblib.dump(imputer,  "features/desc_imputer.pkl")
joblib.dump(selector, "features/desc_selector.pkl")
# Save full descriptor name list (pre-imputer) so predict.py can use the
# exact same descriptor order regardless of RDKit version.
with open("features/desc_names_all.txt", "w") as f:
    f.write("\n".join(desc_names))
with open("features/desc_names.txt", "w") as f:
    f.write("\n".join(selected_desc_names))

print(f"RDKit 描述符（过滤后）: {X_desc.shape}  → features/X_desc.npy")
print(f"  原始描述符数: {X_desc_raw.shape[1]}, 过滤后: {X_desc.shape[1]}")

# ── 2.3 Combined features ─────────────────────────────────────────────────────
X_combined = np.hstack([X_ecfp4, X_desc])
np.save("features/X_combined.npy", X_combined)
print(f"组合特征: {X_combined.shape}  → features/X_combined.npy")
