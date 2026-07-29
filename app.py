import streamlit as st
import os
import time
import uuid
import sqlite3
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, Literal

# --- LangChain & LangGraph 组件 ---
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# 导入自定义模块 (完全保留)
import config
from rag_module import ask_knowledge_base
from LSTM_model_module import run_mill_model_training
from ra_model_module import train_roughness_model
from vision_module import analyze_industrial_image

from universal_module import industrial_universal_diagnostics # 导入新的万能接口
# --- 1. 页面配置与工业风样式 (完全保留) ---
st.set_page_config(
    page_title="InduMind | 工业加工智能 Agent ",
    layout="wide",
    page_icon="🏭",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stStatusWidget { border-radius: 8px; border: 1px solid #d1d8e0; }
    .stChatMessage { border-radius: 12px; }
    .status-box {
        padding: 15px;
        border-radius: 8px;
        background-color: white;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 2. 核心系统状态检查 (完全保留) ---
def check_system_status():
    status = {
        "knowledge_db": os.path.exists(config.DB_DIR),
        "nasa_dataset": os.path.exists(config.CSV_PATH_NASA) and os.path.exists(config.DATA_MILL_DIR),
        "ra_dataset": all(os.path.exists(path) for path in config.BASE_DATA_DIR.values()),
        "ollama_service": False
    }
    try:
        import requests
        res = requests.get("http://localhost:11434/api/tags", timeout=1)
        if res.status_code == 200: status["ollama_service"] = True
    except:
        status["ollama_service"] = False
    return status


# --- 3. LangGraph 定义 (完全保留) ---
TOOL_OPTIONS = {
    "📚 知识库检索": ask_knowledge_base,
    "📈 NASA模型训练": run_mill_model_training,
    "🔧 RA预测训练": train_roughness_model,
    "🔍 工业全自动诊断分析": industrial_universal_diagnostics, # 唯一的数据诊断入口
    "🖼️ 视觉分析": analyze_industrial_image
}

tools_map = {t.name: t for t in TOOL_OPTIONS.values()}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_model(state: AgentState):
    response = st.session_state.llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}


def tool_node_with_ui(state: AgentState):
    result_messages = []
    last_message = state['messages'][-1]
    for tool_call in last_message.tool_calls:
        t_name = tool_call["name"]
        t_args = tool_call["args"]
        with st.status(f"⚙️ 正在执行工业组件: {t_name}...", expanded=True) as status:
            try:
                func = tools_map[t_name]
                output = func.invoke(t_args) if hasattr(func, 'invoke') else func(**t_args)
                status.update(label=f"🎯 {t_name} 执行完毕", state="complete")
            except Exception as e:
                output = f"执行异常: {str(e)}"
                status.update(label=f"❌ {t_name} 失败", state="error")
            result_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call["id"]))
    return {"messages": result_messages}


def should_continue(state: AgentState) -> Literal["tools", END]:
    return "tools" if state['messages'][-1].tool_calls else END


@st.cache_resource
def get_memory_db():
    conn = sqlite3.connect("indumind_storage.db", check_same_thread=False)
    return SqliteSaver(conn)


def create_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node_with_ui)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=get_memory_db())


# --- 4. Streamlit 主界面 ---
def main():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🏭 InduMind AI</h2>", unsafe_allow_html=True)
        st.markdown("---")

        # 长期记忆线选择
        st.subheader("🧠 记忆与档案管理")
        machine_id = st.selectbox(
            "选择任务编号",
            ["CNC-001", "CNC-002", "CNC-003", "Experimental-Lab"],
            help="不同编号拥有独立的长期记忆"
        )
        st.info(f"当前 Thread ID: `{machine_id}`")

        st.markdown("---")
        # --- 修改点：此处由原来的一个 CSV 上传变为两个专用上传口 ---
        st.subheader("📸 实时诊断输入")

        # 统一上传入口
        uploaded_csv = st.file_uploader("上传实时传感器CSV数据", type="csv")
        uploaded_img = st.file_uploader("上传表面形貌图片", type=["jpg", "png"])

        file_context = ""
        if uploaded_csv:
            p = os.path.join(config.TEMP_DATA_DIR, uploaded_csv.name)
            with open(p, "wb") as f: f.write(uploaded_csv.getbuffer())
            file_context += f"\n【系统提示：检测到数据，请调用 industrial_universal_diagnostics 分析 {p}】"
            st.success("数据已就绪")


        st.markdown("---")
        # 系统状态 (保持不变)
        st.subheader("📊 系统实时状态")
        sys_status = check_system_status()

        def status_dot(is_ok):
            return "🟢 正常" if is_ok else "🔴 异常"

        st.markdown(f"""
        <div class="status-box">
        <b>记忆引擎:</b> 🟢 SQLite Persistent<br>
        <b>核心模型:</b> <code>{config.OLLAMA_MODEL_NAME}</code><br>
        <hr style='margin: 10px 0;'>
        <b>知识库:</b> {status_dot(sys_status['knowledge_db'])}<br>
        <b>数据集:</b> {status_dot(sys_status['ra_dataset'])}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ 彻底清空当前记忆", use_container_width=True):
            try:
                import sqlite3
                conn = sqlite3.connect("indumind_storage.db")
                cursor = conn.cursor()
                cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (machine_id,))
                cursor.execute("DELETE FROM writes WHERE thread_id = ?", (machine_id,))
                conn.commit()
                conn.close()
                st.session_state.messages = []
                st.success(f"档案 {machine_id} 已重置")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"清空失败: {str(e)}")

    st.title("工业智能加工 Agent 决策系统")
    st.caption("基于 LangGraph 持久化工作流的切削监测平台")

    # --- 工具选择器 (完全保留) ---
    with st.popover("🛠️ Tools (配置组件)"):
        st.markdown("### 🔌 插件中心")
        active_tools = []
        for label, func in TOOL_OPTIONS.items():
            is_on = st.toggle(label, value=True)
            if is_on:
                active_tools.append(func)

    # --- 模型绑定 (完全保留) ---
    llm = ChatOllama(model=config.OLLAMA_MODEL_NAME, temperature=0.1)
    if active_tools:
        st.session_state.llm_with_tools = llm.bind_tools(active_tools)
    else:
        st.session_state.llm_with_tools = llm

    if "agent_graph" not in st.session_state:
        st.session_state.agent_graph = create_agent_graph()

    # --- 消息同步与显示 (完全保留) ---
    thread_config = {"configurable": {"thread_id": machine_id}}
    if "messages" not in st.session_state:
        st.session_state.messages = []

    graph_state = st.session_state.agent_graph.get_state(thread_config)
    db_messages = graph_state.values.get("messages", [])

    display_messages = []
    for m in db_messages:
        if isinstance(m, HumanMessage):
            display_messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            display_messages.append({"role": "assistant", "content": m.content})

    for msg in display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- 用户输入与执行逻辑 (完全保留并优化) ---
    if prompt := st.chat_input("请输入加工诊断指令..."):
        full_prompt = prompt + file_context  # 组合用户输入和文件路径提示词

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if 'last_training_img' in st.session_state: del st.session_state['last_training_img']

            system_msg = SystemMessage(
                content="你是 InduMind-V3 工业专家。请使用简体中文回复。\n\n"
                        "**你的工业常识更新手册（针对 NASA 数据集）：**\n"
                        "1. **主轴电流 (smcAC)**：\n"
                        "   - 空载状态下 P99 通常为 0.1 - 1.0 A。\n"
                        "   - **正式切削状态下 P99 达到 4.0 - 5.5 A 是完全正常的物理现象**。\n"
                        "   - 只有当 P99 超过 **8.0 A** 或者出现【异常平直的直线信号】（例如 P99 与均值几乎相等，Std 接近 0）时，才判定为传感器故障。\n\n"
                        "2. **振动信号 (vib)**：\n"
                        "   - P99 在 3.0 - 6.5 之间属于正常的机械运作波动。\n\n"
                        "3. **诊断逻辑：**\n"
                        "   - 优先信任【神经网络推理】得出的磨损值（VB）。\n"
                        "   - 只有当物理指纹（P99）出现【断崖式跳变】或【不符合物理规律的超高值（如电流 > 10.0）】时，才在建议中提示数据风险。\n"
                        "   - 如果你发现某列数据是完全平直的（如 Mean=2.3, P99=2.3），即使数值在正常范围内，也要指出这是异常的信号。"
    )


            try:
                with st.spinner(f"🧠 正在推理..."):
                    input_data = {"messages": [HumanMessage(content=full_prompt)]}
                    if not db_messages:
                        input_data["messages"].insert(0, system_msg)

                    final_state = st.session_state.agent_graph.invoke(input_data, config=thread_config)
                    response_text = final_state["messages"][-1].content

                st.markdown(response_text)

                # 自动显示训练图表 (关键逻辑：保留)
                if 'last_training_img' in st.session_state:
                    img_path = st.session_state['last_training_img']
                    if os.path.exists(img_path):
                        st.image(img_path, caption="📈 关联分析结果可视化", use_column_width=True)

            except Exception as e:
                st.error(f"决策引擎异常: {str(e)}")


if __name__ == "__main__":
    main()