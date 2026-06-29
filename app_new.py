import os
import json
import time
import tempfile
import streamlit as st

# 导入 LangChain 核心及混合检索组件
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 引入新版 v1.0 兼容的 Ensemble 检索与 BM25 组件
from langchain_community.retrievers.bm25 import BM25Retriever
# noinspection PyUnresolvedReferences
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatZhipuAI

# ==================== 1. 基础配置与路径定义 ====================
st.set_page_config(page_title="多轮记忆智能知识库系统", page_icon="📚", layout="wide")
st.title("📚 基于 RAG 架构的多轮对话智能知识库系统")

DB_DIR = "./faiss_index"  # 本地向量库存储路径
SESSIONS_DIR = "./chat_sessions"  # 历史多会话目录
METADATA_FILE = "./db_metadata.json"  # 本地已加载文件元数据

# 确保多会话文件夹物理存在
os.makedirs(SESSIONS_DIR, exist_ok=True)


# ==================== 2. 多会话持久化底层函数 ====================
def list_sessions():
    files = [f for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SESSIONS_DIR, x)), reverse=True)
    return [os.path.splitext(f)[0] for f in files]


def load_session(session_id):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"title": "损坏的会话 ⚠️", "messages": []}
    return {"title": "新会话 💬", "messages": []}


def save_session(session_id, title, messages):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"title": title, "messages": messages}, f, ensure_ascii=False, indent=2)


def delete_session(session_id):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)


def save_metadata(uploaded_filenames):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(list(uploaded_filenames), f, ensure_ascii=False, indent=2)


def load_metadata():
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


# ==================== 3. 辅助：格式化历史消息 ====================
def format_chat_history(messages):
    """将前几次的对答历史格式化为纯文本，用于重写 Prompt（仅取最近 5 轮防止 Token 爆炸）"""
    formatted = []
    for msg in messages[-5:]:
        role = "用户" if msg["role"] == "user" else "AI助手"
        # 过滤掉冗长的参考文献数据，只保留对话纯文本
        content = msg["content"]
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


# ==================== 4. 初始化全局状态机 ====================
all_local_sessions = list_sessions()
if not all_local_sessions:
    default_sid = f"session_{int(time.time())}"
    save_session(default_sid, "新会话 💬", [])
    all_local_sessions = [default_sid]

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = all_local_sessions[0]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = load_metadata()


# ==================== 5. 混合检索与向量管道核心 ====================
def build_hybrid_retriever(vector_store):
    chunks = list(vector_store.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2

    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )
    return ensemble_retriever


def append_single_file_to_vector_store(uploaded_file, zhipu_api_key, vector_store):
    os.environ["ZHIPUAI_API_KEY"] = zhipu_api_key
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".txt":
            loader = TextLoader(temp_file_path, encoding="utf-8")
        elif file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        else:
            return vector_store

        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        embeddings = ZhipuAIEmbeddings(model="embedding-3")
        batch_size = 50

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)

        return vector_store
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ==================== 6. 侧边栏：多会话选择与知识管理 ====================
with st.sidebar:
    st.header("🔑 系统配置")
    zhipu_api_key = st.text_input("输入您的智谱 API Key", type="password")

    if st.session_state.vector_store is None and os.path.exists(DB_DIR) and zhipu_api_key:
        try:
            os.environ["ZHIPUAI_API_KEY"] = zhipu_api_key
            embeddings = ZhipuAIEmbeddings(model="embedding-3")
            st.session_state.vector_store = FAISS.load_local(
                folder_path=DB_DIR,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            st.success("✅ 自动从本地恢复了历史知识库！")
        except Exception:
            st.warning("本地知识库恢复失败，可能 API Key 不正确。")

    st.header("💬 会话管理")
    if st.button("➕ 新开会话", use_container_width=True):
        new_sid = f"session_{int(time.time())}"
        save_session(new_sid, "新会话 💬", [])
        st.session_state.current_session_id = new_sid
        st.rerun()

    sessions = list_sessions()
    session_info = {sid: load_session(sid) for sid in sessions}

    try:
        current_idx = sessions.index(st.session_state.current_session_id)
    except ValueError:
        current_idx = 0
        st.session_state.current_session_id = sessions[0]

    selected_sid = st.selectbox(
        "选择或切换会话窗口：",
        options=sessions,
        index=current_idx,
        format_func=lambda x: f"💬 {session_info[x]['title']}"
    )

    if selected_sid != st.session_state.current_session_id:
        st.session_state.current_session_id = selected_sid
        st.rerun()

    if st.button("🗑️ 删除当前会话窗口", use_container_width=True):
        delete_session(st.session_state.current_session_id)
        remaining = list_sessions()
        if remaining:
            st.session_state.current_session_id = remaining[0]
        else:
            fresh_sid = f"session_{int(time.time())}"
            save_session(fresh_sid, "新会话 💬", [])
            st.session_state.current_session_id = fresh_sid
        st.success("会话已删除！")
        st.rerun()

    st.write("---")
    st.header("📂 知识库管理")

    if st.session_state.loaded_files:
        st.markdown("**📂 当前知识库包含文件：**")
        for filename in sorted(list(st.session_state.loaded_files)):
            st.markdown(f"- `{filename}`")
        st.write("---")

    uploaded_files = st.file_uploader(
        "上传新文档（支持增量追加）",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if not zhipu_api_key:
            st.error("⚠️ 请先在上方输入您的智谱 API Key，再执行构建！")
        else:
            new_files = [f for f in uploaded_files if f.name not in st.session_state.loaded_files]
            if new_files:
                if st.button(f"📥 增量导入 {len(new_files)} 个新文档"):
                    with st.spinner("正在增量解析并提取特征中..."):
                        vs = st.session_state.vector_store
                        for f in new_files:
                            vs = append_single_file_to_vector_store(f, zhipu_api_key, vs)
                            st.session_state.loaded_files.add(f.name)

                        st.session_state.vector_store = vs
                        if vs:
                            vs.save_local(DB_DIR)
                            save_metadata(st.session_state.loaded_files)
                            st.success("🎉 增量导入并保存成功！")
                            st.rerun()
            else:
                st.info("ℹ️ 选择的文件均已存在于当前知识库中。")

    if st.button("🔥 彻底销毁知识库"):
        st.session_state.vector_store = None
        st.session_state.loaded_files = set()
        if os.path.exists(DB_DIR):
            import shutil

            shutil.rmtree(DB_DIR)
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)
        st.success("知识库已彻底销毁！")
        st.rerun()

# ==================== 7. 主交互区 ====================
active_session_data = load_session(st.session_state.current_session_id)
active_messages = active_session_data["messages"]
active_title = active_session_data["title"]

if not zhipu_api_key:
    st.info("👈 请先在左侧侧边栏输入您的【智谱 API Key】开始体验。")
elif st.session_state.vector_store is None:
    st.info("💡 当前本地无可用知识库，请在左侧上传并导入文档。")
else:
    # 渲染历史气泡和引用的文献出处
    for msg in active_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander("🔍 展开查看本次回答的参考文献出处"):
                    for i, doc in enumerate(msg["sources"]):
                        st.markdown(f"**文献片段 {i + 1}** (来自文件: `{doc['source']}`):")
                        st.code(doc['content'], language="markdown")
                        st.write("---")

    if user_query := st.chat_input("请输入您对知识库的疑问..."):
        # 1. 记录并渲染用户输入
        active_messages.append({"role": "user", "content": user_query})

        # 实时根据首问修改会话标题
        if len(active_messages) == 1 or active_title == "新会话 💬":
            active_title = user_query[:15] + ("..." if len(user_query) > 15 else "")

        save_session(st.session_state.current_session_id, active_title, active_messages)

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("正在检索多文档并组织回答..."):
                os.environ["ZHIPUAI_API_KEY"] = zhipu_api_key
                llm = ChatZhipuAI(model="glm-4-flash", temperature=0)

                # ==================== 🛠️ 核心升级：历史问题重写链 ====================
                # 如果当前会话已经有历史记录，启动大模型重写问题
                # 排除我们刚刚 append 进来的最新用户提问，提取之前的纯历史记录
                history_messages = active_messages[:-1]

                if len(history_messages) > 0:
                    formatted_history = format_chat_history(history_messages)

                    rephrase_system_prompt = (
                        "根据给出的【对答历史】和【用户的最新提问】，将用户的最新提问重写为一个独立的、完整的、包含所有具体背景名词的搜索问题。\n"
                        "这个搜索问题将直接用于向量数据库检索。请保持提问的客观和完整，不要回答提问，仅输出重写后的独立问题。如果最新提问本身就很独立完整，则直接原样输出。"
                    )
                    rephrase_prompt = ChatPromptTemplate.from_messages([
                        ("system", rephrase_system_prompt),
                        ("human", "【对答历史】:\n{history}\n\n【最新提问】:\n{question}")
                    ])
                    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

                    # 运行重写链，获取去指代消解后的搜索词
                    search_query = rephrase_chain.invoke({"history": formatted_history, "question": user_query})
                else:
                    # 如果是会话的第一句，直接使用原问题作为搜索词
                    search_query = user_query
                # ====================================================================

                # 使用重写后的 search_query 触发双路混合检索
                hybrid_retriever = build_hybrid_retriever(st.session_state.vector_store)
                retrieved_docs = hybrid_retriever.invoke(search_query)

                context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                # 大模型最终回答 Prompt：融入检索内容与对话历史，保证代词（那、它、其）语义绝对连贯
                system_prompt = (
                    "你是一个专业且温馨的企业知识库助手。\n"
                    "请结合下方提供的【背景资料】和【对答历史】，回答用户最新提出的问题。\n\n"
                    "【答题原则】：\n"
                    "1. 允许并鼓励进行符合生活常识的合理推理（例如：晚上9点属于20:00之后；'晚饭钱'、'加班餐'可等同于'餐饮补贴'）。\n"
                    "2. 回答时请直接给出结论和具体规定（如金额、流程等），并保持语气自然、专业。\n"
                    "3. 如果【背景资料】中确实没有提及与问题相关的任何线索，再回答：'抱歉，根据公司规定守则，我没有找到相关信息。'\n\n"
                    "【背景资料】:\n{context}\n\n"
                    "【对答历史】:\n{history_text}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{question}")
                ])

                chain = prompt | llm | StrOutputParser()

                # 拼接刚才格式化好的对话历史发送给最终的问答链
                history_text_to_send = format_chat_history(history_messages) if len(history_messages) > 0 else "无历史记录"
                ai_response = chain.invoke({
                    "context": context,
                    "history_text": history_text_to_send,
                    "question": user_query
                })

                # 提取参考文献
                sources_data = [
                    {"source": doc.metadata.get("source", "未知文件"), "content": doc.page_content}
                    for doc in retrieved_docs
                ]

                # 展示大模型生成的回答
                st.markdown(ai_response)

                # 并在 expander 里展示重写后的检索问题（这个作为 Debug 细节极其惊艳，面试加分项）
                with st.expander("🔍 展开查看本次回答的参考文献出处"):
                    st.info(f"💡 [RAG 多轮指代消解优化] 本次检索使用的独立重写问题为: `{search_query}`")
                    st.write("---")
                    for i, doc in enumerate(retrieved_docs):
                        # 使用正统的 LangChain Document 对象属性方式访问：
                        source_name = doc.metadata.get("source", "未知文件")
                        st.markdown(f"**文献片段 {i + 1}** (来自文件: `{source_name}`):")
                        st.code(doc.page_content, language="markdown")
                        st.write("---")

                # 保存 AI 回答、参考文献到本地
                active_messages.append({
                    "role": "assistant",
                    "content": ai_response,
                    "sources": sources_data
                })

                save_session(st.session_state.current_session_id, active_title, active_messages)
                st.rerun()