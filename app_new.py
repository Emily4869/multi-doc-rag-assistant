import os
import tempfile
import streamlit as st

# 导入核心组件
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==================== 1. 基础配置与会话状态初始化 ====================
st.set_page_config(page_title="多文档智能知识库系统", page_icon="📚", layout="wide")
st.title("📚 基于 RAG 架构的多文档智能知识库系统")

# 初始化全局会话状态
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================== 2. 后端文档处理函数（支持多文件） ====================
def process_uploaded_files(uploaded_files, zhipu_api_key):
    """循环接收多个上传文件，解析、切片，并分批写入 FAISS 向量库（解决智谱 64 条上限限制）"""
    all_chunks = []

    # 临时配置当前调用所需的 API KEY
    os.environ["ZHIPUAI_API_KEY"] = zhipu_api_key

    for uploaded_file in uploaded_files:
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
                st.warning(f"跳过不支持的文件: {uploaded_file.name}")
                continue

            documents = loader.load()

            # 记录文件来源元数据
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    if not all_chunks:
        return None

    # ==================== 🛠️ 核心修复逻辑：分批构建数据库 ====================
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    vector_store = None

    # 每次最多发送 50 条切片给智谱，保证绝对安全不超限（阈值 64）
    batch_size = 50

    for i in range(0, len(all_chunks), batch_size):
        # 截取当前批次的切片
        batch = all_chunks[i: i + batch_size]

        if vector_store is None:
            # 第一批：初始化创建 FAISS 数据库
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            # 后续批次：增量添加进现有的数据库中
            vector_store.add_documents(batch)

    # ====================================================================

    return vector_store


# ==================== 3. 页面布局：侧边栏（安全秘钥与多文件上传） ====================
with st.sidebar:
    st.header("🔑 系统配置")
    # 安全输入框（用户输入的 Key 仅临时保存在页面会话中，不会上传至任何服务器，安全可靠）
    zhipu_api_key = st.text_input("输入您的智谱 API Key", type="password")

    st.header("📂 知识库文档管理")
    # 开启 accept_multiple_files=True 从而支持一次上传多个文件
    uploaded_files = st.file_uploader(
        "上传企业文档 (支持多个 .txt, .pdf 文件)",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    # 构建按钮逻辑触发
    if uploaded_files:
        if not zhipu_api_key:
            st.error("⚠️ 请先在上方输入您的智谱 API Key，再执行构建！")
        else:
            if st.session_state.vector_store is None or st.button("🔄 重新构建合并索引"):
                with st.spinner(f"正在对 {len(uploaded_files)} 个文档计算向量并构建索引..."):
                    st.session_state.vector_store = process_uploaded_files(uploaded_files, zhipu_api_key)
                    if st.session_state.vector_store:
                        st.success(f"🎉 成功构建包含了 {len(uploaded_files)} 个文档的联合知识库！")
                        st.session_state.messages = []

# ==================== 4. 页面主体：Chat 交互区 ====================
if not zhipu_api_key:
    st.info("👈 请先在左侧侧边栏输入您的【智谱 API Key】开始体验。")
elif st.session_state.vector_store is None:
    st.info("💡 请在左侧侧边栏上传一个或多个企业文档，系统将自动构建专属联合知识库。")
else:
    st.subheader("💬 专属智能多文档 AI 对话")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("请输入关于您上传的多文档的任意问题..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("正在检索多文档并组织回答..."):
                os.environ["ZHIPUAI_API_KEY"] = zhipu_api_key
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(user_query)

                context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                system_prompt = (
                    "你是一个专业且温馨的企业知识库助手。\n"
                    "请结合下方提供的【背景资料】，回答用户的问题。\n\n"
                    "【答题原则】：\n"
                    "1. 允许并鼓励进行符合生活常识的合理推理（例如：晚上9点属于20:00之后；'晚饭钱'、'加班餐'可等同于'餐饮补贴'）。\n"
                    "2. 回答时请直接给出结论和具体规定（如金额、流程等），并保持语气自然、专业。\n"
                    "3. 如果【背景资料】中确实没有提及与问题相关的任何线索，再回答：'抱歉，根据公司规定守则，我没有找到相关信息。'\n\n"
                    "【背景资料】:\n{context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{question}")
                ])

                llm = ChatZhipuAI(model="glm-4-flash", temperature=0)
                chain = prompt | llm | StrOutputParser()
                ai_response = chain.invoke({"context": context, "question": user_query})

                st.markdown(ai_response)

                # 引用来源展示（展示具体的来源文件名）
                with st.expander("🔍 展开查看本次回答的参考文献出处（支持跨文件回溯）"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**文献片段 {i + 1}** (来自文件: `{doc.metadata.get('source')}`):")
                        st.code(doc.page_content, language="markdown")
                        st.write("---")

                st.session_state.messages.append({"role": "assistant", "content": ai_response})