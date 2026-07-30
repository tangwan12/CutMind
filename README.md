<h1 align="center">🧠 CutMind</h1>
<h3 align="center">工业加工智能 Agent 决策系统</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.0.39-green" alt="LangGraph">
  <img src="https://img.shields.io/badge/Streamlit-1.32-red" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## 📋 概述

**CutMind** 是一个面向工业机床加工场景的智能诊断 Agent 平台。它通过 LangGraph 编排工作流，将 LSTM 刀具磨损预测、CNN 表面粗糙度评估、RAG 知识库检索、工业视觉分析和通用传感器诊断整合为一个可交互的智能系统。

用户只需上传传感器数据或工件照片，通过自然语言与 Agent 交互，即可获得刀具磨损量、表面加工质量等关键指标的诊断结论。

---
'''
## 🔧 核心功能

| 功能模块 | 说明 | 技术栈 |
|---------|------|--------|
| **刀具磨损预测** | 基于 NASA 铣削数据集，用 LSTM 预测刀具磨损量 VB | PyTorch LSTM |
| **表面粗糙度预测** | 根据 17 通道传感器数据预测加工表面粗糙度 Ra | PyTorch 1D-CNN |
| **RAG 知识库检索** | 检索工业切削标准、加工机理文档 | ChromaDB + HuggingFace Embeddings |
| **工业视觉分析** | 分析工件照片，评估表面质量 | ShuffleNet (TorchVision) |
| **通用传感器诊断** | 自动匹配传感器数据到对应推理模型 | 多模型注册表 |
| **对话式交互** | 基于 LangGraph 的 ReAct Agent 工作流 | Ollama + LangChain |
| **持久化记忆** | 按任务编号（CNC-001 等）独立存储对话历史 | SQLite + LangGraph Checkpointer |

---
'''
## 🏗️ 架构图

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Web UI                    │
│    ┌──────────────┐  ┌──────────────────────────┐   │
│    │  侧边栏      │  │       主聊天区域          │   │
│    │  · 任务编号  │  │  · 用户输入 / 文件上传    │   │
│    │  · 文件上传  │  │  · Agent 对话 & 诊断结果  │   │
│    │  · 系统状态  │  │  · 训练可视化图表        │   │
│    └──────────────┘  └──────────────────────────┘   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│          LangGraph Agent 工作流                       │
│                                                       │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│   │  Agent   │───▶│  Tools   │───▶│  Agent   │      │
│   │ (思考)   │    │ (执行)   │    │ (回答)   │      │
│   └──────────┘    └──────────┘    └──────────┘      │
│        │                │                             │
│   ┌────┴────┐    ┌──────┴──────┐                     │
│   │  Ollama  │    │  工具节点    │                     │
│   │ qwen3:8b │    │ (5 个组件)  │                     │
│   └─────────┘    └─────────────┘                     │
└──────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                    工具层                              │
│                                                       │
│  ┌─────────┐ ┌──────┐ ┌───────┐ ┌──────┐ ┌───────┐ │
│  │ 知识库  │ │ LSTM │ │ CNN   │ │ 视觉 │ │ 万能  │ │
│  │ 检索    │ │ 磨损 │ │ 粗糙度│ │ 分析 │ │ 诊断  │ │
│  └─────────┘ └──────┘ └───────┘ └──────┘ └───────┘ │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- [Ollama](https://ollama.com/)（本地大模型推理引擎）
- 建议 16GB+ 内存用于模型推理

### 安装步骤

# 1. 克隆仓库
git clone https://github.com/tangwan12/CutMind.git
cd CutMind

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载并启动 Ollama 模型
ollama pull qwen3:8b

# 4. 准备数据（可选，用于训练）
 - NASA 铣削数据集放入 data_mill/ 目录
 - 机床加工数据放入 new_machine_data1212/ 等目录
 - 工业 PDF 文档放入 paper/ 目录（自动构建知识库）

# 5. 启动
streamlit run app.py
`

---

## 📁 项目结构

`
CutMind/
├── app.py                     # 主程序入口 — Streamlit + LangGraph
├── config.py                  # 全局配置（模型路径、数据路径等）
├── requirements.txt           # Python 依赖
├── .gitignore                 # Git 忽略规则
│
├── LSTM_model_module.py       # NASA 刀具磨损 LSTM 预测模型
├── ra_model_module.py         # 表面粗糙度 CNN 预测模型
├── rag_module.py              # RAG 知识库检索（ChromaDB + PDF）
├── vision_module.py           # 工业视觉分析（ShuffleNet）
├── universal_module.py        # 通用传感器诊断入口
│
├── paper/                     # 工业技术参考文档（PDF）
├── chroma_db_langchain/       # 向量数据库（运行后自动生成）
├── data_mill/                 # NASA 铣削数据集（需自行下载）
├── new_machine_data1212/      # 机床加工数据（需自行准备）
├── new_machine_data1219/      # 机床加工数据（需自行准备）
├── temp_data/                 # 上传文件临时存储
└── uploads/                   # 上传文件存储
`

---

## 💡 使用方式

1. **启动应用**：streamlit run app.py
2. **选择任务编号**：在侧边栏选择 CNC-001 / CNC-002 / CNC-003 / Experimental-Lab，每个编号拥有独立的对话记忆
3. **上传数据**（可选）：上传 CSV 传感器数据或 JPG/PNG 工件照片
4. **提问诊断**：在聊天框输入自然语言指令，例如：
   - "分析这批数据，看看刀具磨损是否正常"
   - "对比 CNC-001 和 CNC-002 的磨损趋势"
   - "训练磨损预测模型"
   - "查一下切削参数对表面粗糙度的影响"
5. **查看结果**：Agent 会自动调度合适的工具进行分析，并返回诊断结论和可视化图表

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **前端** | Streamlit |
| **Agent 框架** | LangChain, LangGraph |
| **大模型** | Ollama (qwen3:8b / 可切换) |
| **深度学习** | PyTorch (LSTM, CNN) |
| **视觉** | TorchVision (ShuffleNet) |
| **向量检索** | ChromaDB, HuggingFace Embeddings |
| **持久化** | SQLite |
| **数据处理** | Pandas, NumPy, Scikit-learn |

---

## 📌 注意事项

- 首次运行需要下载 Ollama 模型（qwen3:8b 约 4.7GB）
- 启用工具前，确保对应数据集已就位
- RAG 知识库需将 PDF 文档放入 paper/ 目录，首次启动时自动构建
- 本项目的 Earth-Agent-main/ 子目录为 [Earth-Agent](https://github.com/opendatalab/Earth-Agent) 遥感项目，不包含在仓库中

---

## 📝 License

MIT License © 2026 tangwan12
