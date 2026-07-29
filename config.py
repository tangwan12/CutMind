import os

# Ollama 配置
OLLAMA_MODEL_NAME = "qwen3:8b"

# RAG 配置
MODEL_PATH = r"E:\yan\RAG\models\paraphrase-multilingual-MiniLM-L12-v2"
PDF_DIR = "paper"
DB_DIR = "./chroma_db_langchain"

# --- 1. 刀具磨损模型 (VB) 配置 ---
CSV_PATH_NASA = "mill.csv"
DATA_MILL_DIR = "data_mill"
MODEL_WEAR_SAVE = "LSTM_model.pth"
IMG_WEAR_PATH = "LSTM_result.png"
DEFAULT_EPOCHS_LSTM = 40

# --- 2. 表面粗糙度模型 (Ra) 配置 ---
BASE_DATA_DIR = {
    "12.12": "new_machine_data1212",
    "12.19": "new_machine_data1219"
}
MODEL_RA_SAVE = "ra_cnn_model.pth"
IMG_RA_PATH = "ra_result.png"
DEFAULT_EPOCHS_RA = 20
SAMPLE_COUNT_RA = 1200
BATCH_SIZE_RA = 32

# 【新增/修改】强制规定输入通道数，确保训练和诊断完全一致
# 根据报错，你的模型是按 17 个通道训练的
RA_INPUT_CHANNELS = 17

# --- 3. 多模态与诊断专用配置 ---
UPLOAD_DIR = "uploads"
TEMP_DATA_DIR = "temp_data"
for d in [UPLOAD_DIR, TEMP_DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

RA_THRESHOLD = 0.05