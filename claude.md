# CYP11B2 / CYP11B1 QSAR 建模项目

## 项目背景

针对醛固酮合酶（CYP11B2）抑制剂的选择性预测建模项目。
核心挑战是 CYP11B2 与 CYP11B1 之间 93% 的序列同源性，需要通过模型辅助筛选对 CYP11B2 有活性、对 CYP11B1 选择性强的化合物。

## 数据文件

- `data.txt`：原始活性数据，制表符分隔，三列：
  - 第一列：SMILES
  - 第二列：CYP11B2 IC50（μM）
  - 第三列：CYP11B1 IC50（μM）
- 数据来源：专利公开数据，全部为正的精确数值（已排除截断值和 n.d.）
- 数值为 0 的条目含义不明确，需在数据清洗阶段丢弃

## 建模指南

详细步骤、代码和注意事项见 `CYP11B_QSAR_modeling_guide.md`，执行任何建模任务前必须先完整阅读该文件。

## 建模策略

- 建立两个**独立的回归模型**：Model_B2 预测 pIC50(CYP11B2)，Model_B1 预测 pIC50(CYP11B1)
- 不建选择性模型（训练数据缺乏选择性阴性样本）
- 联合使用两个模型：B2 预测值高 + B1 预测值低 = 优先化合物
- 活性单位统一换算为 pIC50，公式：`pIC50 = 6 - log10(IC50_uM)`

## 输出规范

- 清洗后数据：`data_cleaned.csv`
- 模型文件：`models/` 目录（.pkl 格式）
- 图表和评估结果：`results/` 目录
- 每个步骤生成独立的 Python 脚本文件，不使用 notebook

## 技术栈

```
rdkit, pandas, numpy, scikit-learn, xgboost, lightgbm, deepchem, shap, matplotlib, scipy
```

## 执行原则

- 遇到报错自行调试修复，无需中断询问
- 优先使用 scaffold split 评估模型，随机划分仅作对比参考
- 所有模型均需输出：R²、RMSE、MAE、Pearson r 以及 parity plot
