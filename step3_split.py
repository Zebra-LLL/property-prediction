"""
Step 3: Dataset Splitting
Inputs:  data_cleaned.csv
Outputs: splits/scaffold_split.npz  (train_idx, val_idx, test_idx)
         splits/random_split.npz    (train_idx, val_idx, test_idx)
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

os.makedirs("splits", exist_ok=True)

df = pd.read_csv("data_cleaned.csv")
total = len(df)
print(f"化合物数: {total}")

# ── 3.1 Scaffold Split (primary evaluation) ───────────────────────────────────
def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

print("计算 Bemis-Murcko 骨架...")
df["scaffold"] = df["smiles_std"].apply(get_scaffold)

scaffold_to_indices = defaultdict(list)
for i, scaffold in enumerate(df["scaffold"]):
    scaffold_to_indices[scaffold].append(i)

# Sort: largest scaffold groups first (assigned to training)
scaffolds_sorted = sorted(scaffold_to_indices.items(),
                          key=lambda x: len(x[1]), reverse=True)

train_idx, val_idx, test_idx = [], [], []

for scaffold, indices in scaffolds_sorted:
    if len(test_idx) / total < 0.1:
        test_idx.extend(indices)
    elif len(val_idx) / total < 0.1:
        val_idx.extend(indices)
    else:
        train_idx.extend(indices)

train_idx = np.array(sorted(train_idx))
val_idx   = np.array(sorted(val_idx))
test_idx  = np.array(sorted(test_idx))

print(f"Scaffold split → 训练: {len(train_idx)}, 验证: {len(val_idx)}, 测试: {len(test_idx)}")
print(f"  比例 → 训练: {len(train_idx)/total:.1%}, 验证: {len(val_idx)/total:.1%}, 测试: {len(test_idx)/total:.1%}")

np.savez("splits/scaffold_split.npz",
         train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
print("已保存 splits/scaffold_split.npz")

# Save scaffold column back for reference
df.to_csv("data_cleaned.csv", index=False)

# ── 3.2 Random Split (for comparison baseline) ────────────────────────────────
idx_all = np.arange(total)
idx_train_r, idx_test_r = train_test_split(idx_all, test_size=0.2, random_state=42)
idx_train_r, idx_val_r  = train_test_split(idx_train_r, test_size=0.125, random_state=42)
# Approx train:val:test = 70:10:20

print(f"\nRandom split → 训练: {len(idx_train_r)}, 验证: {len(idx_val_r)}, 测试: {len(idx_test_r)}")

np.savez("splits/random_split.npz",
         train_idx=idx_train_r, val_idx=idx_val_r, test_idx=idx_test_r)
print("已保存 splits/random_split.npz")

# ── Summary ───────────────────────────────────────────────────────────────────
n_scaffolds = len(scaffold_to_indices)
print(f"\n独特骨架数: {n_scaffolds}  (覆盖率 {n_scaffolds/total:.1%})")
print("注：Scaffold split 的 R² 将显著低于随机划分，以 scaffold 结果为准。")
