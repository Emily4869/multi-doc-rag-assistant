现在，我们迎来了整个项目的收官之战——**第五步：搭建 Streamlit 可视化网页界面**。

在这一步中，我们将把前面做好的所有硬核后端逻辑（解析、切片、FAISS 检索、提示词优化、GLM 问答）整合到一个漂亮的 Web 网页中，实现一个类似于 **ChatGPT + PDF 智能助手** 的完整产品。

用户可以在网页上：
1. **上传**本地任意 PDF 或 TXT 文档。
2. 系统自动在后台进行“解析-切片-向量化-存入 FAISS”的一键初始化。
3. 提供一个精致的**聊天对话框**，让用户直接提问，并展示**检索出的原文出处（Source）**。

---

### 💡 核心原理剖析

#### 1. 为什么用 Streamlit 搭建 AI 前端？
在真实的 AI 项目原型开发或企业内部工具开发中，写 HTML/CSS/JS 的成本太高，且不利于快速迭代。
**Streamlit** 是目前大模型时代最主流的 Python 网页框架。它能让你用纯 Python 代码，在几十分钟内写出非常美观、支持响应式布局的 Web 界面，完美契合大模型应用的交互。

#### 2. Streamlit 的运行机制与“状态丢失”痛点（面试常问）
Streamlit 的运行原理非常独特：**每次用户和页面发生交互（如点击按钮、输入文字），Streamlit 都会将整个 `.py` 文件从头到尾重新运行一遍。**
*   *痛点：* 这会导致普通的 Python 变量在每次运行后被全部销毁。如果不做处理，用户每发送一句话，我们切分好的向量数据库和历史聊天记录都会被清空并重新初始化。这既浪费时间，又耗费 API 额度。
*   *解决：* 我们必须利用 **`st.session_state`（会话状态）** 来存储跨越多次运行、需要持久保存的全局变量（如：`st.session_state.vector_store` 和 `st.session_state.messages`）。

---

### 🛠️ 动手实践

#### 1. 安装 Streamlit 依赖
在你的虚拟环境中安装 streamlit：
```bash
pip install streamlit
```

#### 2. 编写全套整合脚本 `app.py`
在你的项目文件夹下，新建一个名为 `app.py` 的文件，写入以下完整的、工程级交互代码：

```python
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
os.environ["ZHIPUAI_API_KEY"] = "你的_ZHIPU_API_KEY"

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
                        st.markdown(f"**文献片段 {i+1}：**")
                        st.code(doc.page_content, language="markdown")
                        st.write("---")

                # 4.6 保存 AI 回复到历史纪录中
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
```

#### 3. 运行你的网页应用
在你的终端里运行以下命令（确保在项目虚拟环境下，且在 `app.py` 所在目录）：
```bash
streamlit run app.py
```

终端运行后，**系统会自动在你的浏览器里弹出一个网页**（通常是 `http://localhost:8501`）。

---

### 📝 你的最终调试任务

现在，你可以亲自扮演用户，体验这个完整的 AI 产品：
1. 打开网页，按照提示，在左侧上传我们之前准备的 `sample_company_policy.txt`，看看是否会提示解析成功。
2. 在右侧聊天框输入：“我加班到晚上十点能报销晚饭钱吗？有发票要求吗？” 看看 AI 能不能回答出来。
3. 展开底部的“🔍 展开查看本次回答的参考文献出处”，看看里面找出来的片段是不是正确的。
4. **进阶挑战：** 找一份你自己的本地 PDF 文档（比如一页产品说明、简历、或者几页的技术手册），上传到页面中，尝试向它提问一些细节。看看它是否能同样准确地进行问答！

当你完成这最后一步，在浏览器中流畅地与你亲手写出的应用进行对话时，你就已经**从头到尾、完整且有深度地交付了你的第一个 RAG 项目**！

运行成功后请告诉我，我们进行最后的成果大集结，教你如何将这些丰富的工程收获写在简历里！