"""
Predict pIC50_B2 (CYP11B2) and pIC50_B1 (CYP11B1) for new compounds.

Usage
-----
# Single SMILES:
python predict.py --smiles "CCc1ccc(NC(=O)c2ccc(F)cc2)cc1"

# Batch from file (one SMILES per line, optional name column):
python predict.py --input compounds.smi

# Batch from CSV (must have a 'smiles' column):
python predict.py --input compounds.csv --smiles_col smiles

# Save results to CSV:
python predict.py --input compounds.smi --output predictions.csv

Required files (relative to this script's directory)
------------------------------------------------------
models/feature_selector.pkl   — IndexSelector (utils.IndexSelector)
models/final_model_B2.pkl     — Pipeline(StandardScaler + SVR)
models/final_model_B1.pkl     — Pipeline(StandardScaler + SVR)
features/desc_imputer.pkl     — fitted median imputer for RDKit descriptors
features/desc_selector.pkl    — fitted VarianceThreshold selector
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import joblib

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

from utils import IndexSelector   # required for feature_selector.pkl

# ── Load saved artefacts ───────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def _load(path):
    full = os.path.join(BASE, path)
    if not os.path.exists(full):
        sys.exit(f"[ERROR] Required file not found: {full}")
    return joblib.load(full)

feature_selector = _load("models/feature_selector.pkl")
model_b2         = _load("models/final_model_B2.pkl")
model_b1         = _load("models/final_model_B1.pkl")
desc_imputer     = _load("features/desc_imputer.pkl")
desc_selector    = _load("features/desc_selector.pkl")

# ── Feature calculators (must match step2_features.py exactly) ────────────────
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_desc_names = [name for name, _ in Descriptors.descList]
_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(_desc_names)


def smiles_to_features(smiles_list):
    """
    Compute X_combined (n, 2189) for a list of SMILES strings.
    Invalid SMILES are replaced with zero vectors / median-imputed descriptors.
    """
    ecfp4_rows, desc_rows = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi).strip()) if pd.notna(smi) else None

        # ECFP4
        if mol is not None:
            fp = _morgan_gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
        else:
            fp = np.zeros(2048, dtype=np.uint8)
        ecfp4_rows.append(fp)

        # RDKit 2D descriptors
        if mol is not None:
            desc = list(_calculator.CalcDescriptors(mol))
        else:
            desc = [np.nan] * len(_desc_names)
        desc_rows.append(desc)

    X_ecfp4 = np.vstack(ecfp4_rows)
    X_desc_raw = np.array(desc_rows, dtype=float)

    # Apply the same imputer + variance selector fitted on training data
    X_desc = desc_selector.transform(desc_imputer.transform(X_desc_raw))

    X_combined = np.hstack([X_ecfp4, X_desc])          # (n, 2189)
    X_selected = feature_selector.transform(X_combined)  # (n, 300)
    return X_selected


def predict(smiles_list):
    """Return DataFrame with columns: smiles, pIC50_B2, pIC50_B1."""
    X = smiles_to_features(smiles_list)
    pred_b2 = model_b2.predict(X)
    pred_b1 = model_b1.predict(X)
    return pd.DataFrame({
        "smiles":    smiles_list,
        "pIC50_B2":  pred_b2.round(4),
        "pIC50_B1":  pred_b1.round(4),
    })


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Predict CYP11B2/B1 pIC50 for new compounds"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles",     type=str, help="Single SMILES string")
    group.add_argument("--input",      type=str, help="Input file (.smi or .csv)")
    parser.add_argument("--smiles_col", type=str, default="smiles",
                        help="Column name for SMILES in CSV (default: smiles)")
    parser.add_argument("--output",    type=str, default=None,
                        help="Output CSV path (default: print to stdout)")
    args = parser.parse_args()

    # ── Collect SMILES ─────────────────────────────────────────────────────────
    if args.smiles:
        smiles_list = [args.smiles]

    else:
        path = args.input
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Input file not found: {path}")

        if path.endswith(".csv"):
            df_in = pd.read_csv(path)
            if args.smiles_col not in df_in.columns:
                sys.exit(f"[ERROR] Column '{args.smiles_col}' not found in {path}. "
                         f"Available columns: {list(df_in.columns)}")
            smiles_list = df_in[args.smiles_col].tolist()
        else:
            # .smi or plain text: first whitespace-delimited token per line
            with open(path) as f:
                smiles_list = [line.split()[0] for line in f if line.strip()]

    # ── Predict ────────────────────────────────────────────────────────────────
    results = predict(smiles_list)

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Saved {len(results)} predictions → {args.output}")
    else:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
