import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from langchain_core.tools import tool
import config


# ==========================================
# 1. 定义 LSTM 模型结构 (完全保留)
# ==========================================
class LSTM(nn.Module):
    def __init__(self, sensor_dim=9000, extra_dim=4, hidden_dim=64):
        super(LSTM, self).__init__()
        self.sensor_feature_extract = nn.Sequential(
            nn.Linear(sensor_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.extra_fc = nn.Sequential(
            nn.Linear(extra_dim, 16),
            nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, sensor_list, extra_data):
        combined_sensors = torch.stack(sensor_list, dim=1)
        sensor_features = self.sensor_feature_extract(combined_sensors)
        lstm_out, _ = self.lstm(sensor_features)
        last_time_step = lstm_out[:, -1, :]
        extra_features = self.extra_fc(extra_data)
        cat_features = torch.cat((last_time_step, extra_features), dim=1)
        prediction = self.regressor(cat_features)
        return prediction


# ==========================================
# 2. 全量训练工具 (完全保留 100% 原始逻辑)
# ==========================================
@tool
def run_mill_model_training(epochs: int = 40) -> str:
    """
    【技能名称】：NASA 数据集 LSTM 磨损预测模型训练。
    用于从零开始训练 NASA 全量数据集。包含数据清洗、特征对齐与深度训练。
    如果用户不要求更改训练轮次，请用设定好的 40 作为训练轮次。
    """
    status_container = st.status(f"🚀 正在启动全流程训练任务 (Epochs={epochs})...", expanded=True)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        status_container.write(f"💻 训练设备: {device}")

        if not os.path.exists(config.CSV_PATH_NASA) or not os.path.exists(config.DATA_MILL_DIR):
            return "错误：找不到数据文件。请检查 mill.csv 和 data_mill 文件夹。"

        status_container.write("📂 正在读取数据并清洗...")
        df = pd.read_csv(config.CSV_PATH_NASA)
        nan_rows_index = list(map(str, df[df['VB'].isna()].index.tolist()))
        df = df.dropna(subset=['VB'])
        df = df[df['VB'] > 0.01]

        data_list = df['VB'].tolist()
        ex_attribute = df[['DOC', 'feed', 'material']].values.tolist()
        case_list = df['case'].tolist()
        vb_data_list = np.array(data_list, dtype=np.float32)

        first_case = []
        for i in range(len(case_list)):
            if i > 0 and case_list[i] != case_list[i - 1]:
                first_case.append(str(i))

        smcAC, smcDC, vib_t, vib_s, ae_t, ae_s = [], [], [], [], [], []
        ALL_lists = [smcAC, smcDC, vib_t, vib_s, ae_t, ae_s]
        data_names = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']

        files = os.listdir(config.DATA_MILL_DIR)
        progress_bar = status_container.progress(0, text="读取传感器 CSV...")

        valid_count = 0
        for idx, filename in enumerate(files):
            if filename.endswith('.csv'):
                base = os.path.splitext(filename)[0]
                if base in nan_rows_index or base in first_case: continue
                filepath = os.path.join(config.DATA_MILL_DIR, filename)
                sub_df = pd.read_csv(filepath)
                for i in range(6):
                    d = sub_df[data_names[i]].tolist()
                    d = d[:9000] if len(d) > 9000 else d + [0] * (9000 - len(d))
                    ALL_lists[i].append(d)
                valid_count += 1
            if idx % 20 == 0: progress_bar.progress(min(idx / len(files), 1.0))

        training_data = []
        min_len = min(len(vb_data_list) - 1, valid_count)
        for i in range(min_len):
            prev_vb = vb_data_list[i - 1] if (i > 0 and case_list[i] == case_list[i - 1]) else 0.0
            ex_feat = torch.FloatTensor(ex_attribute[i] + [prev_vb]).to(device)
            sensors = [torch.FloatTensor(ALL_lists[k][i]).to(device) for k in range(6)]
            training_data.append((sensors, ex_feat, vb_data_list[i]))

        train_set, test_set = train_test_split(training_data, test_size=0.2, random_state=42)
        model = LSTM(sensor_dim=9000, extra_dim=4).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_bar = status_container.progress(0, text="Training...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for s_list, ex_feat, target in train_set:
                optimizer.zero_grad()
                s_list_batched = [s.unsqueeze(0) for s in s_list]
                output = model(s_list_batched, ex_feat.unsqueeze(0))
                loss = criterion(output, torch.tensor([[target]], device=device))
                loss.backward();
                optimizer.step();
                epoch_loss += loss.item()
            train_bar.progress((epoch + 1) / epochs, text=f"Epoch {epoch + 1}/{epochs}")

        torch.save(model.state_dict(), config.MODEL_WEAR_SAVE)
        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for s_list, ex_feat, target in test_set:
                s_list_batched = [s.unsqueeze(0) for s in s_list]
                out = model(s_list_batched, ex_feat.unsqueeze(0))
                preds.append(out.item());
                targs.append(target)
        r2 = r2_score(targs, preds)
        plt.figure(figsize=(10, 5))
        plt.plot(targs, 'b-', label='Actual');
        plt.plot(preds, 'r-', label='Predicted')
        plt.title(f"Validation R2: {r2:.4f}");
        plt.legend();
        plt.savefig(config.IMG_WEAR_PATH);
        plt.close()
        st.session_state['last_training_img'] = config.IMG_WEAR_PATH
        status_container.update(label="✅ 训练完成", state="complete")
        return f"✅ 成功！R²: {r2:.4f}"
    except Exception as e:
        status_container.update(label="❌ 失败", state="error")
        return f"错误: {str(e)}"



def run_wear_inference_engine(df: pd.DataFrame) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM().to(device)
    model.load_state_dict(torch.load(config.MODEL_WEAR_SAVE, map_location=device))
    model.eval()

    sensor_tensors = []
    # 按照 NASA 标准取 6 列
    for i in range(6):
        d = df.iloc[:, i].values.tolist()
        d = d[:9000] if len(d) > 9000 else d + [0] * (9000 - len(d))
        sensor_tensors.append(torch.FloatTensor(d).unsqueeze(0).to(device))

    extra = torch.FloatTensor([1.5, 0.25, 1.0, 0.1]).unsqueeze(0).to(device)
    with torch.no_grad():
        res = model(sensor_tensors, extra).item()
    return f"{res:.4f} mm (磨损量VB)"