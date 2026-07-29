import os
import torch
import pandas as pd
import numpy as np
from langchain_core.tools import tool
import config
# 动态导入模型推理函数
from LSTM_model_module import run_wear_inference_engine
from ra_model_module import run_ra_inference_engine

# ==========================================
# 1. 工业模型注册表 (未来增加模型只需在此添加配置)
# ==========================================
MODEL_REGISTRY = {
    "nasa_wear": {
        "keywords": ['smcAC', 'smcDC', 'vib_table', 'AE_table', 'AE_spindle'],
        "channels": 6,
        "weight_path": config.MODEL_WEAR_SAVE,
        "desc": "NASA 铣刀磨损监测模型",
        "inference_func": run_wear_inference_engine
    },
    "ra_roughness": {
        "keywords": ['AI1-01', 'AI2-01'],
        "channels": 17,
        "weight_path": config.MODEL_RA_SAVE,
        "desc": "RA 表面粗糙度预测模型",
        "inference_func": run_ra_inference_engine
    }
}


@tool
def industrial_universal_diagnostics(csv_path: str) -> str:
    """【工业感官】：提取 CSV 数据中的原始物理特征，不进行任何主观评估。"""
    if not os.path.exists(csv_path): return "❌ 文件未找到。"
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        # 预处理：只计算最客观的指纹
        stats_md = "| 传感器列名 | 均值(Mean) | 峰值(P99) | 波动性(Std) |\n| :--- | :--- | :--- | :--- |\n"
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                vals = df[col].dropna().values
                stats_md += f"| {col} | {np.mean(vals):.4f} | {np.percentile(vals, 99):.4f} | {np.std(vals):.4f} |\n"

        # 推理逻辑（仅提供数值）
        ai_reports = []
        for task, cfg in MODEL_REGISTRY.items():
            if len(df.columns) == cfg['channels'] or any(k in str(df.columns).lower() for k in cfg['keywords']):
                if os.path.exists(cfg['weight_path']):
                    res = cfg['inference_func'](df)

                    ai_reports.append(f"- **{cfg['desc']} 预测数值**: `{res}`")

        # 最终返回：只给数据事实，不给结论
        return (f"### 📡 传感器观测原始快照\n"
                f"- **分析文件**: `{os.path.basename(csv_path)}`\n"
                f"#### 1. 物理层指纹扫描:\n{stats_md}\n"
                f"#### 2. 神经网络推理输出:\n"
                f"{chr(10).join(ai_reports) if ai_reports else '- 无匹配AI模型'}\n"
                f"\n**[注意]**：请大模型专家结合物理指纹和预测值，自主判定当前设备是否存在异常。")
    except Exception as e:
        return f"感官组件异常: {str(e)}"