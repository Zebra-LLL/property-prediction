#!/usr/bin/env bash
# CYP11B2/B1 QSAR 建模全流程
# 使用前确保 data.txt 已放置于当前目录
# 执行方式: bash run_pipeline.sh

set -e  # 任意步骤失败则退出

echo "======================================================"
echo " CYP11B2 / CYP11B1 QSAR 建模流程"
echo "======================================================"

# 检查数据文件
if [ ! -f "data.txt" ]; then
    echo "错误：data.txt 不存在，请将原始数据放置于当前目录"
    exit 1
fi

mkdir -p models results features splits

echo ""
echo "Step 1: 数据预处理..."
python step1_preprocess.py

echo ""
echo "Step 2: 特征计算..."
python step2_features.py

echo ""
echo "Step 3: 数据集划分..."
python step3_split.py

echo ""
echo "Step 4: 模型训练 (RF + XGBoost, B2 + B1)..."
python step4_train.py

echo ""
echo "Step 5: 模型评估与图表生成..."
python step5_evaluate.py

echo ""
echo "Step 6: SHAP 可解释性分析..."
python step6_shap.py

echo ""
echo "======================================================"
echo " 流程完成！"
echo " 清洗数据 → data_cleaned.csv"
echo " 模型文件 → models/"
echo " 图表结果 → results/"
echo "======================================================"
