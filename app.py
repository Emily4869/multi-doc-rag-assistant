import os
import tempfile
import streamlit as st

# 导入前几步我们验证成功的核心组件
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==================== 1. 基础配置与会话状态初始化 ====================
st.set_page_config(page_title="企业知识库智能问答助手", page_icon="🤖", layout="wide")
st.title("🤖 基于 RAG 架构的企业文档智能问答系统")

# 配置你的智谱 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPU_API_KEY")
# 使用 st.session_state 初始化全局变量，防止每次页面刷新后丢失数据
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # 存储加载好的 FAISS 向量库
if "messages" not in st.session_state:
    st.session_state.messages = []  # 存储聊天历史记录


# ==================== 2. 后端文档处理函数 ====================
def process_uploaded_file(uploaded_file):
    """接收上传的文件，保存到临时目录，并构建 FAISS 向量数据库"""
    # 2.1 将 Streamlit 上传的内存文件保存为本地临时文件，以便 LangChain Loader 读取
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    try:
        # 2.2 根据文件后缀，自动选择合适的加载器
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".txt":
            loader = TextLoader(temp_file_path, encoding="utf-8")
        elif file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        else:
            st.error("暂不支持该文件格式，请上传 .txt 或 .pdf 文件")
            return None

        documents = loader.load()

        # 2.3 文档切片
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
        chunks = text_splitter.split_documents(documents)

        # 2.4 构建 FAISS 向量库并存入全局 session_state
        embeddings = ZhipuAIEmbeddings(model="embedding-3")
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    finally:
        # 清理临时文件，保持系统干净
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ==================== 3. 页面布局：侧边栏（文档上传） ====================
with st.sidebar:
    st.header("📂 知识库文档管理")
    uploaded_file = st.file_uploader("上传企业文档(支持 .txt, .pdf)", type=["txt", "pdf"])

    if uploaded_file:
        # 只有在向量库未加载，或者用户上传了新文件时，才重新构建向量库
        if st.session_state.vector_store is None or st.button("🔄 重新解析并构建索引"):
            with st.spinner("正在解析文档并计算向量，请稍候..."):
                st.session_state.vector_store = process_uploaded_file(uploaded_file)
                if st.session_state.vector_store:
                    st.success(f"🎉 '{uploaded_file.name}' 解析成功，知识库就绪！")
                    # 清空上一份文件的历史聊天记录
                    st.session_state.messages = []

# ==================== 4. 页面主体：Chat 交互区 ====================
# 如果用户还没有上传并构建向量库，给予温馨提示
if st.session_state.vector_store is None:
    st.info("💡 请先在左侧侧边栏上传您的企业守则、PDF文档或文本文件，系统将为您自动构建专属的智能问答服务。")
else:
    st.subheader("💬 专属智能 AI 对话")

    # 4.1 渲染历史聊天信息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 4.2 捕捉用户输入
    if user_query := st.chat_input("请输入您想咨询的文档内容..."):
        # 立即展示用户的提问
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 4.3 执行 RAG 检索与大模型推理
        with st.chat_message("assistant"):
            # 这里的 st.spinner 能在等待 AI 回复时展示旋转特效，提升用户体验
            with st.spinner("正在深度阅读文档并组织回答..."):
                # a. 执行检索 (k=3)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                retrieved_docs = retriever.invoke(user_query)

                # b. 拼接上下文
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                # c. 优化后的 RAG 提示词
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

                # d. 调用大模型生成
                llm = ChatZhipuAI(model="glm-4-flash", temperature=0)

                # 组装纯 LangChain 链并运行
                chain = prompt | llm | StrOutputParser()
                ai_response = chain.invoke({"context": context, "question": user_query})

                # 4.4 渲染并展示大模型的回答
                st.markdown(ai_response)

                # 4.5 展示可折叠的【参考文献来源】── 这是 RAG 的核心体验，让答案有据可查
                with st.expander("🔍 展开查看本次回答的参考文献出处"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**文献片段 {i + 1}：**")
                        st.code(doc.page_content, language="markdown")
                        st.write("---")

                # 4.6 保存 AI 回复到历史纪录中
                st.session_state.messages.append({"role": "assistant", "content": ai_response})