"""
Step 1: Data Preprocessing
- Read data.in (SMILES + IC50_B2 + IC50_B1)
- Parse and clean activity values (remove 0, >, <, n.d.)
- Convert IC50 (uM) to pIC50
- Standardize SMILES (desalt, uncharge, normalize)
- Deduplicate by InChIKey (take mean pIC50)
- Output: data_cleaned.csv
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.inchi import MolToInchiKey

# Instantiate once (loading salt list per-call is expensive and fragile)
_normalizer = rdMolStandardize.Normalizer()
_uncharger   = rdMolStandardize.Uncharger()
_chooser     = rdMolStandardize.LargestFragmentChooser()

# ── 1. Read raw data ──────────────────────────────────────────────────────────
df = pd.read_csv("data.in", sep="\t", header=0,
                 names=["smiles", "IC50_B1_uM", "IC50_B2_uM"])
print(f"原始行数: {len(df)}")

# ── 2. Parse activity values ──────────────────────────────────────────────────
def parse_activity(val):
    """
    Returns float or NaN.
    NaN cases: missing, >, <, n.d., or 0 (ambiguous meaning).
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.startswith(">") or s.startswith("<") or s.lower() == "n.d.":
        return np.nan
    try:
        v = float(s)
        return np.nan if v == 0 else v
    except ValueError:
        return np.nan

df["IC50_B2_uM"] = df["IC50_B2_uM"].apply(parse_activity)
df["IC50_B1_uM"] = df["IC50_B1_uM"].apply(parse_activity)

# Convert to pIC50: pIC50 = 6 - log10(IC50_uM)
df["pIC50_B2"] = df["IC50_B2_uM"].apply(
    lambda x: 6 - np.log10(x) if pd.notna(x) and x > 0 else np.nan
)
df["pIC50_B1"] = df["IC50_B1_uM"].apply(
    lambda x: 6 - np.log10(x) if pd.notna(x) and x > 0 else np.nan
)

print(f"B2 有效数据: {df['pIC50_B2'].notna().sum()}")
print(f"B1 有效数据: {df['pIC50_B1'].notna().sum()}")

# ── 3. Standardize SMILES ─────────────────────────────────────────────────────
def standardize_smiles(smi):
    """Keep largest fragment, normalize, uncharge, return canonical SMILES."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = _chooser.choose(mol)      # desalt via largest fragment
        if mol is None:
            return None
        mol = _normalizer.normalize(mol)
        if mol is None:
            return None
        mol = _uncharger.uncharge(mol)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None

df["smiles_std"] = df["smiles"].apply(standardize_smiles)
invalid = df["smiles_std"].isna().sum()
print(f"无效 SMILES: {invalid} 个，已丢弃")
df = df.dropna(subset=["smiles_std"]).reset_index(drop=True)

# ── 4. Deduplicate by InChIKey (mean pIC50) ───────────────────────────────────
def get_inchikey(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return MolToInchiKey(mol)

df["inchikey"] = df["smiles_std"].apply(get_inchikey)

df_dedup = df.groupby("inchikey", as_index=False).agg(
    smiles_std=("smiles_std", "first"),
    pIC50_B2=("pIC50_B2", "mean"),
    pIC50_B1=("pIC50_B1", "mean"),
)
print(f"去重后化合物数: {len(df_dedup)}")

# ── 5. Save cleaned data ──────────────────────────────────────────────────────
df_dedup.to_csv("data_cleaned.csv", index=False)
print("已保存 data_cleaned.csv")

# Print pIC50 distribution summary
for col in ["pIC50_B2", "pIC50_B1"]:
    valid = df_dedup[col].dropna()
    print(f"\n{col} 分布 (n={len(valid)}):")
    print(f"  min={valid.min():.2f}, max={valid.max():.2f}, "
          f"mean={valid.mean():.2f}, std={valid.std():.2f}")
