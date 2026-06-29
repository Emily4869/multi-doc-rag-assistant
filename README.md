# 📚 Enterprise Multi-Doc RAG Evolutionary Tutorial
### 基于 LangChain 与 Streamlit 的多文档智能问答系统（从零到一进化指南）

这是一个循序渐进的 RAG（检索增强生成）系统开发教程。本项目不仅包含了最终交付的 **Streamlit 多文档智能问答系统** 源码，还完整记录了从“Hello World”一步步演进到“企业级应用”的完整开发心路历程，包括在 Windows 环境下的 **C++ 内存越界崩溃排查** 和 **大模型防御性拒答调优** 等核心工程实践。

---

## 📂 项目进化阶段与目录结构

本项目代码按照由浅入深的原则进行组织，你可以按照 `steps/` 目录下的脚本顺序，观察一个 RAG 系统的进化过程：

```text
my-rag-assistant/
├── steps/                           # 💡 核心开发进化阶段
│   ├── step1_simple_call.py         # 阶段 1：理解 ChatModel 抽象接口与 LCEL 运行机制
│   ├── step2_doc_chunking.py        # 阶段 2：理解文档解析（Loader）与递归切片（Splitter）
│   ├── step3_vector_store.py        # 阶段 3：向量化（Embedding）与向量存储（FAISS 引擎）
│   └── step4_rag_chain.py           # 阶段 4：RAG 完整链组装与提示词防幻觉约束优化
├── app.py                           # 🚀 阶段 5 (最终交付)：多文档 Web 交互系统 (Streamlit)
├── sample_company_policy.txt        # 测试用企业守则文档
├── requirements.txt                 # 项目环境依赖
└── .gitignore                       # Git 忽略配置
```

---

## 🛠️ 项目逐步进化详解与底层原理

### 🚀 阶段 1：最简调用与 LCEL 机制 (`steps/step1_simple_call.py`)
- **核心原理解析：** 
  - **ChatModel 与传统 LLM 的区别：** 传统 LLM 仅接收纯字符串，而 ChatModel 接收并输出标准的结构化消息列表（Message List），包括 `SystemMessage`（设定）、`HumanMessage`（用户）和 `AIMessage`（模型），更符合现代大模型微调对齐的接口。
  - **LCEL（表达式语言）工作原理：** 利用 Python 的 `__or__` 魔法方法重载竖线 `|` 运算符，将 `prompt | model | parser` 串联成一个实现了 `Runnable` 协议的管道流，让输入数据自动流转。

---

### 🚀 阶段 2：文档切片与边界重叠策略 (`steps/step2_doc_chunking.py`)
- **核心原理解析：** 
  - **为何切片：** 解决大模型的上下文长度限制，控制 API 调用成本，避免超长文本导致的“注意力丢失 (Lost in the Middle)”现象。
  - **切片策略：** 采用 `RecursiveCharacterTextSplitter` 按照双换行、单换行、空格等优先级递归切分，尽量保证语义段落完整。
  - **边界重叠（Overlap）：** 通过 `chunk_overlap=30` 在相邻片段之间保留重叠内容，防止一句话被物理切断时导致语义丢失。

---

### 🚀 阶段 3：向量存储与系统级 Bug 排查 (`steps/step3_vector_store.py`)
- **核心原理解析：** 
  - **Embedding 机制：** 将文本映射为多维空间坐标，使得含义相近的文本具有更高的余弦相似度（Cosine Similarity）。
  - **向量数据库：** 传统关系型数据库只支持字面硬匹配，向量数据库支持超高维向量的空间距离近似搜索（ANN）。
- **🚨 踩坑记录 (Bug 001)：**
  - **现象：** 使用 ChromaDB 运行时，进程无任何报错，直接以状态码 `-1073741819 (0xC0000005)` 异常闪退。
  - **原因：** `0xC0000005` 是 Windows 的内存访问违规（段错误）。ChromaDB 底层的 C++ 核心库 `hnswlib` 在编译时与部分 Windows 环境或 CPU 指令集产生硬件冲突，导致内核强制关闭进程。
  - **工程解决：** 充分利用 LangChain 对向量数据库的高抽象特性，通过修改 2 行代码无缝切换至稳定兼容的工业级检索库 **FAISS**（Facebook AI Similarity Search），问题得以解决。

---

### 🚀 阶段 4：RAG 完整链与提示词微调 (`steps/step4_rag_chain.py`)
- **核心原理解析：** 
  - **知识注入：** 检索出来的相关文本作为背景（Context）动态拼入 Prompt，约束大模型“只能基于已知背景作答”，消除幻觉。
- **🚨 踩坑记录 (Bug 002)：**
  - **现象：** 检索出了正确文本（包含餐饮报销40元），但问“加班到九点报销晚饭钱吗”时，轻量级模型依然防御性回答“没有找到相关规定”。
  - **原因：** **提示词过度约束（Prompt Over-Constraint）**。为了追求防幻觉，提示词使用了强否定性语气（如“绝对禁止编造”），导致轻量级模型（如 glm-4-flash）极度保守，拒绝进行基础常识推理（如：晚上九点属于20:00之后、晚饭钱就是餐饮补贴）。
  - **工程解决：** 优化 Prompt，显式建立**“常识推理授权区”**，主动向模型授权语义等价映射的边界，在防幻觉与泛化解答之间取得了更好的平衡。

---

### 🚀 阶段 5 (最终交付)：多文档 Web 交互网页 (`app.py`)
- **核心原理解析：** 
  - **状态持久化：** 使用 Streamlit 的 `st.session_state` 保存向量库和聊天记录，防止页面刷新重新加载而引发数据丢失。
  - **联合数据库构建：** 支持一次上传多个 PDF/TXT，并将它们在内存中统一解析并聚合成一个 FAISS 数据库。
  - **密钥安全防泄露：** 提供侧边栏密码式 API Key 输入框，拒绝代码硬编码，确保代码可以安全公开部署。

---

## ⚙️ 快速开始运行

### 1. 克隆并进入项目
```bash
git clone <你的GitHub仓库链接>
cd my-rag-assistant
```

### 2. 创建并激活虚拟环境
```bash
python -m venv env
# Windows 激活
env\Scripts\activate
# macOS/Linux 激活
source env/bin/activate
```

### 3. 安装依赖包
```bash
pip install -r requirements.txt
```

### 4. 启动最终网页应用
```bash
streamlit run app.py
```
启动成功后，在浏览器中打开 `http://localhost:8501`，配置你的智谱 API Key，即可体验多文档智能问答助手！
```

---

### 🚀 开始推送到 GitHub

1. **重新导出依赖包（确保包含 streamlit、faiss-cpu 和 pypdf 等）：**
   在终端运行：
   ```bash
   pip freeze > requirements.txt
   ```
2. **确认 `.gitignore` 包含以下过滤项，避免大文件及本地数据库被错传：**
   ```text
   env/
   venv/
   __pycache__/
   *.pyc
   faiss_index/
   chroma_db/
   ```
3. **在终端依次执行上传指令：**
   ```bash
   git init
   git add .
   git commit -m "feat: complete step-by-step RAG evolutionary project with streamlit UI"
   git branch -M main
   git remote add origin <你的GitHub远程仓库链接>
   git push -u origin main
   ```

上传完成后，刷新你的 GitHub 页面。你将看到一个设计考究、不仅展示最终成果还展示了扎实排错思维的 RAG 核心开源项目。无论哪个面试官看到这样的仓库，都会对你的工程能力和学习态度留下极其深刻的印象！


