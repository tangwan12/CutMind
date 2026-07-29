import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
import config

@st.cache_resource(show_spinner="正在初始化向量数据库...")
def get_vector_store():
    if os.path.exists(config.MODEL_PATH):
        embedding = HuggingFaceEmbeddings(model_name=config.MODEL_PATH)
    else:
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    db_exists = os.path.exists(config.DB_DIR) and len(os.listdir(config.DB_DIR)) > 0
    vector_store = Chroma(persist_directory=config.DB_DIR, embedding_function=embedding)

    if not db_exists:
        if os.path.exists(config.PDF_DIR):
            loader = DirectoryLoader(config.PDF_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
            raw_docs = loader.load()
            if raw_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
                docs = text_splitter.split_documents(raw_docs)
                vector_store.add_documents(docs)
    return vector_store

@tool
def ask_knowledge_base(query: str) -> str:
    """【技能名称】：工业理论知识库检索。用于回答切削机理、加工标准等专业问题。"""
    try:
        vector_store = get_vector_store()
        results = vector_store.as_retriever(search_kwargs={"k": 3}).invoke(query)
        if not results: return "知识库中未找到相关内容。"
        context = "\n\n".join([doc.page_content for doc in results])
        return f"【检索到的参考资料】：\n{context}"
    except Exception as e:
        return f"检索错误: {str(e)}"