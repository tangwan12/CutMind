import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from langchain_core.tools import tool
import config


# ==========================================
# 1. 模型定义
# ==========================================
class RaPredictorCNN(nn.Module):
    def __init__(self, input_channels, seq_length):
        super(RaPredictorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(128 + 1, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )

    def forward(self, x_sensor, x_prev_ra):
        cnn_out = self.features(x_sensor)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        combined = torch.cat((cnn_out, x_prev_ra), dim=1)
        out = self.regressor(combined)
        return out


# ==========================================
# 2. 核心算法逻辑
# ==========================================
def calculate_ra_logic(segment, poly_degree=1):
    x = np.arange(len(segment))
    if len(segment) <= poly_degree: return 0.0
    coeffs = np.polyfit(x, segment, poly_degree)
    trend = np.polyval(coeffs, x)
    roughness = segment - trend
    return np.mean(np.abs(roughness))


# ==========================================
# 3. 数据预处理（含17通道强制对齐逻辑）
# ==========================================
def prepare_dataset_full(sensor_path, roughness_path, target_samples=1000):
    df_sensor = pd.read_csv(sensor_path)
    X_raw = df_sensor.iloc[:, 1:].values  # 跳过时间轴

    # --- 通道强制对齐逻辑 ---
    current_channels = X_raw.shape[1]
    required_channels = config.RA_INPUT_CHANNELS
    if current_channels > required_channels:
        X_raw = X_raw[:, :required_channels]
    elif current_channels < required_channels:
        padding = np.zeros((X_raw.shape[0], required_channels - current_channels))
        X_raw = np.hstack([X_raw, padding])
    # -----------------------

    try:
        y_raw = pd.read_csv(roughness_path, header=None).values.flatten()
    except:
        y_raw = pd.read_csv(roughness_path).iloc[:, 0].values.flatten()

    ratio = len(X_raw) / len(y_raw)
    r_window_size = 50
    step_size = max(1, int((len(y_raw) - r_window_size) / target_samples))

    X_sensor_list, X_prev_ra_list, y_labels = [], [], []
    prev_ra_value = calculate_ra_logic(y_raw[0:r_window_size])

    count = 0
    for r_start in range(step_size, len(y_raw) - r_window_size, step_size):
        r_segment = y_raw[r_start:r_start + r_window_size]
        current_ra = calculate_ra_logic(r_segment)
        s_start, s_end = int(r_start * ratio), int((r_start + r_window_size) * ratio)
        s_segment = X_raw[s_start:s_end]
        fixed_len = int(r_window_size * ratio)

        if len(s_segment) > fixed_len:
            s_segment = s_segment[:fixed_len]
        elif len(s_segment) < fixed_len:
            pad = np.zeros((fixed_len - len(s_segment), required_channels))
            s_segment = np.vstack([s_segment, pad])

        X_sensor_list.append(s_segment.T)
        X_prev_ra_list.append(prev_ra_value)
        y_labels.append(current_ra)
        prev_ra_value = current_ra
        count += 1
        if count >= target_samples: break

    return (np.array(X_sensor_list, dtype=np.float32),
            np.array(X_prev_ra_list, dtype=np.float32).reshape(-1, 1),
            np.array(y_labels, dtype=np.float32).reshape(-1, 1))


# ==========================================
# 4. 训练技能（恢复指标计算逻辑）
# ==========================================
@tool
def train_roughness_model(folder_date: str, file_id: str) -> str:
    """
    【核心训练工具】：用于分析和训练指定日期的工业加工数据。
    参数要求（严格遵守）：
    1. folder_date: 必须是简写日期格式，例如 '12.12' 或 '12.19'。禁止输入年份。
    2. file_id: 仅输入编号，例如 's1600f1.2-10-1'。禁止包含 '.csv' 后缀。
    """

    # --- 【新增：自动清洗逻辑】 ---
    # 1. 如果 AI 传了 '2023-12-12' 或 '2023.12.12'，自动变成 '12.12'
    if "-" in folder_date:
        parts = folder_date.split("-")
        folder_date = f"{parts[-2]}.{parts[-1]}"  # 取最后两个部分拼成 MM.DD
    elif "2023." in folder_date:
        folder_date = folder_date.replace("2023.", "")

    # 2. 如果 AI 传了 's1600...csv'，自动去掉后缀
    file_id = file_id.replace(".csv", "")
    # ---------------------------

    status = st.status(f"🚀 正在定位数据: {folder_date}/{file_id}", expanded=True)
    try:
        # 现在 folder_date 已经是 '12.12' 这种格式了，可以从 config 里拿到路径
        folder_path = config.BASE_DATA_DIR.get(folder_date)

        if not folder_path:
            return f"错误：找不到日期为 {folder_date} 的数据配置。目前仅支持: {list(config.BASE_DATA_DIR.keys())}"

        csv_path = os.path.join(folder_path, f"{file_id}.csv")

        # 后面原有的逻辑保持不变...
        # last_num = file_id.split('-')[-1]
        # label_name = f"第一批{last_num}.txt" if folder_date == "12.12" else f"{last_num}.txt"
        # ... (此处省略你原有的训练代码)


        last_num = file_id.split('-')[-1]
        csv_path = os.path.join(folder_path, f"{file_id}.csv")
        label_name = f"第一批{last_num}.txt" if folder_date == "12.12" else f"{last_num}.txt"
        label_path = os.path.join(folder_path, label_name)

        X_s, X_p, y = prepare_dataset_full(csv_path, label_path, target_samples=config.SAMPLE_COUNT_RA)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_size = int(0.8 * len(y))

        # 转换为 Tensor
        X_s_train, X_p_train, y_train = torch.from_numpy(X_s[:train_size]), torch.from_numpy(
            X_p[:train_size]), torch.from_numpy(y[:train_size])
        X_s_test, X_p_test, y_test = torch.from_numpy(X_s[train_size:]), torch.from_numpy(
            X_p[train_size:]), torch.from_numpy(y[train_size:])

        train_loader = DataLoader(TensorDataset(X_s_train, X_p_train, y_train), batch_size=config.BATCH_SIZE_RA,
                                  shuffle=True)

        model = RaPredictorCNN(input_channels=config.RA_INPUT_CHANNELS, seq_length=X_s.shape[2]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 训练循环
        for epoch in range(config.DEFAULT_EPOCHS_RA):
            model.train()
            for b_s, b_p, b_y in train_loader:
                b_s, b_p, b_y = b_s.to(device), b_p.to(device), b_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(b_s, b_p), b_y)
                loss.backward();
                optimizer.step()

        # --- 恢复：评估指标计算 ---
        model.eval()
        with torch.no_grad():
            y_pred = model(X_s_test.to(device), X_p_test.to(device)).cpu().numpy()
            y_true = y_test.numpy()

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 保存模型
        torch.save(model.state_dict(), config.MODEL_RA_SAVE)

        # --- 恢复：详细绘图 ---
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='Actual Ra', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Predicted Ra', color='red', linestyle='--')
        plt.title(f"Ra Prediction Results ({folder_date})")
        plt.xlabel("Sample Index")
        plt.ylabel("Roughness (μm)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(config.IMG_RA_PATH)
        plt.close()

        st.session_state['last_training_img'] = config.IMG_RA_PATH
        status.update(label="✅ 训练与评估完成", state="complete")

        return (f"### 📊 模型训练评估报告\n"
                f"- **训练数据集**: {folder_date} / {file_id}\n"
                f"- **输入通道**: {config.RA_INPUT_CHANNELS} (固定)\n"
                f"- **平均绝对误差 (MAE)**: `{mae:.6f}`\n"
                f"- **均方误差 (MSE)**: `{mse:.6f}`\n"
                f"- **决定系数 (R²)**: `{r2:.4f}`\n\n"
                f"模型已优化并保存至 `{config.MODEL_RA_SAVE}`。可视化曲线已生成。")

    except Exception as e:
        status.update(label="❌ 训练异常", state="error")
        return f"错误: {str(e)}"



def run_ra_inference_engine(df: pd.DataFrame) -> str:
    """被万能接口调用，执行 RA 粗糙度神经网络推理"""
    if not os.path.exists(config.MODEL_RA_SAVE): return "RA模型未部署"
    try:
        X_s, X_p = prepare_dataset_full(df, target_samples=200)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RaPredictorCNN(input_channels=config.RA_INPUT_CHANNELS, seq_length=50).to(device)
        model.load_state_dict(torch.load(config.MODEL_RA_SAVE, map_location=device))
        model.eval()
        with torch.no_grad():
            preds = model(torch.from_numpy(X_s).to(device), torch.from_numpy(X_p).to(device))
            res = preds.mean().item()
        return f"{res:.4f} μm (表面粗糙度Ra)"
    except Exception as e:
        return f"RA推理失败: {str(e)}"