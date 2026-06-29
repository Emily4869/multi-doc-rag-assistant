# 📚 Multi-Doc RAG Assistant (基于 RAG 架构的多文档智能问答系统)

本项目基于 **LangChain** 最新 LCEL 语法与 **Streamlit**，结合智谱 AI 大模型研发。支持在 Web 页面一键上传多个 PDF/TXT 文档，并利用向量数据库实现本地语义级的高效、精准智能问答服务。

---

## ✨ 核心特性
- **支持多格式、多文档上传：** 支持一键上传多个 `.pdf` 和 `.txt` 格式的企业内部文档。
- **语义相似度检索：** 底层采用 **FAISS** (Facebook AI Similarity Search) 向量数据库与智谱 `embedding-3` 进行高性能相似度计算。
- **严谨防幻觉设计：** 精心设计 Prompt 工程，通过系统级强约束限制大模型仅基于背景资料回答，对文档外内容表现出高度的严谨性和拒答力。
- **追溯可查：** 支持展开查看回答对应的各文献片段来源（支持跨文档文件名回溯）。
- **用户安全友好：** 支持用户在前端自行配置个人的智谱 API Key，拒绝硬编码，保障秘钥安全。

---

## 🛠️ 安装运行指引

### 1. 克隆本项目
```bash
git clone <你的GitHub仓库地址>
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
### 4. 启动应用
```bash
streamlit run app.py
```
启动成功后，在浏览器中打开 http://localhost:8501 即可进行体验。

# 📚 Enterprise Multi-Doc RAG Evolutionary Project
### 基于 LangChain v1.x 与 Streamlit 的增量多会话智能知识库系统（V1 ➔ V2 演进版）

这是一个完全模拟真实企业开发生命周期的 RAG（检索增强生成）系统。本项目不仅提供了最终交付的 **V2.0 生产级问答系统** 源码，还完整保留了 **V1.0 基础验证版本** 的源码与技术演进日志，用以复现和对比 RAG 系统在真实业务落地时面临的各种瓶颈及优化过程。

---

## 📂 项目进化阶段与目录结构

本项目代码按照由浅入深的原则进行组织，展示了 RAG 系统从基础原型到工业级高级架构的演进过程：

```text
my-rag-assistant/
├── steps/                           # 💡 核心开发进化阶段 (V1.0 筑基部分)
│   ├── step1_simple_call.py         # 阶段 1：ChatModel 接口与 LCEL 最简调用
│   ├── step2_doc_chunking.py        # 阶段 2：文本加载（Loader）与切片（Splitter）
│   ├── step3_vector_store.py        # 阶段 3：向量化与本地存储（FAISS 引擎及 0xC0000005 闪退排查）
│   └── step4_rag_chain.py           # 阶段 4：RAG 完整链组装与提示词防防御性拒答微调
├── app_new.py                           # 🚀 最终交付：V2.0 增量多会话智能知识库系统 (Streamlit)
├── sample_company_policy.txt        # 测试用企业守则文档
├── requirements.txt                 # 项目环境依赖（包含新版 rank_bm25、langchain-classic 等）
└── .gitignore                       # Git 忽略配置（已过滤 faiss_index/ 与 chat_sessions/ 敏感目录）
```

---

## 🛠️ 项目版本演进蓝图 (Version Evolution)

| 维度 | 📦 V1.0 基础验证版 (MVP) | 🚀 V2.0 工业级生产版 (Enterprise) |
| :--- | :--- | :--- |
| **文档导入** | 单次全量覆盖上传，不支持追加新文件 | **增量式物理追加（add_documents）**，支持文件滚雪球式累加 |
| **检索召回率** | 纯语义向量检索，对数字、特定条款易偏离 | **Dense-Sparse 双路混合检索（FAISS 0.6 + BM25 0.4）** |
| **多轮对话能力** | 纯单轮问答，多轮代词指代（如“它、那、怎么报销”）易检索失效 | **历史敏感问题重写（Query Condensation）**，多轮指代消解 |
| **会话历史留存** | 内存变量（session_state），网页刷新/关闭即完全清空 | **多会话本地沙盒（JSON 持久化）**，支持 ChatGPT 式会话切换 |
| **参考文献追溯** | 临时渲染，刷新或页面 Rerun 即完全闪退消失 | **历史参考文献存盘**，每个历史气泡永久且独立绑定出处链接 |
| **API 稳定性** | 上传大文档或多文档，易触发 API 64 条输入上限崩溃 | **滑动窗口分批增量构建算法**，单批 50 条抗压分包写入 |

---

## 💡 V2.0 核心架构进阶与工程调优细节

### 1. 混合检索（Dense-Sparse Hybrid Search）
在 V1.0 的纯语义检索中，系统对特定的行业专用词、合同数字条款（如“第四条”、“1214”、“Win + L”）检索准确率偏低。V2.0 引入了基于经典词频匹配算法的 **BM25 检索器（稀疏检索）**。
通过 LangChain 的 `EnsembleRetriever`，将 **FAISS 语义搜索（权重 0.6）** 与 **BM25 词频匹配（权重 0.4）** 进行加权融合。该架构兼顾了高维语义理解与专有名词的极速精准匹配。

### 2. 多轮对话指代消解与问题重写（Query Condensation）
V1.0 中追问“那怎么报销？”会导致向量数据库检索退化。V2.0 构建了**历史敏感问题重写链（History-Aware Query Re-writer）**。在检索发生前，系统先调用轻量级 LLM，输入 `[前几轮历史会话]` + `[当前模糊追问]`，重写为一个独立的、包含背景名词的完整问句再进行检索，解决了多轮对话退化。

### 3. 多会话持久化与首包重命名（Session State Persistence）
系统在本地建立 `chat_sessions/` 目录进行多会话隔离：
- **首包重命名：** 新建会话时，默认命名为 `"新会话 💬"`。当用户发送**第一句话**时，系统自动截取前 15 个字作为这个会话的标题并更新侧边栏，模拟 ChatGPT 体验。
- **历史文献存盘：** 将每次检索到的文献出处随消息一并序列化。在切换或恢复历史会话窗口时，**每个回答气泡下方依然完好保留它当时引用的参考文献出处**，实现结果永久可核查。

---

## 🚨 核心排错日志（面试高频加分项）

- **Bug 001：Windows 环境下 ChromaDB 内存访问违规（闪退崩溃）**
  * *现象与原因：* 写入向量时 Python 进程以状态码 `0xC0000005`（Windows段错误）闪退。ChromaDB 依赖的 C++ 库 `hnswlib` 编译版与 Windows 运行时库发生内存访问冲突。
  * *解决：* 利用 LangChain 接口抽象，无损重构并迁移至 **FAISS 向量库**，彻底攻克平台兼容性 Bug。

- **Bug 002：强防幻觉约束下的“防御性拒答”**
  * *现象与原因：* 检索正确但问“加班到九点报销晚饭钱吗”时轻量模型生硬拒答。原因在于提示词过度约束（“绝对禁止编造”），导致小参数模型面对恐吓拒绝了基础推理。
  * *解决：* 优化 Prompt 显式建立 **“常识推理授权区”**，主动向模型授权语义等价映射的边界，提升了系统召回表现。

- **Bug 003：API 批量写入超限（Zhipu Code 1214）**
  * *现象与原因：* 一次性上传多份文档时接口报错：`input数组最大不得超过64条`。
  * *解决：* 构建了**滑动窗口分批增量索引算法**，以 50 条切片为单次请求安全阈值进行分包，首批 `from_documents` 初始化，后续循环调用 `add_documents` 追加，彻底保障了多文档的兼容性。

- **Bug 004：LangChain v1.0.0 主版本升级后命名空间缺失**
  * *现象与原因：* 执行 `from langchain.retrievers import ...` 时抛出找不到 `retrievers`。原因为 LangChain 1.x 对核心包进行大规模解耦，传统检索器被剥离至新伴生库 **`langchain-classic`** 中。
  * *解决：* 安装 `langchain-classic` 并在代码中采用最新标准路径导入 `EnsembleRetriever`。

---

## ⚙️ 快速开始运行

### 1. 安装依赖包
```bash
pip install -r requirements.txt
```

### 2. 启动 V2.0 多会话网页应用
在终端运行：
```bash
python -m streamlit run app.py
```
启动成功后，在左侧侧边栏配置你的智谱 API Key，即可体验专属的增量多会话智能知识库系统！
```

---

### 第四部分：上传到 GitHub

请在项目终端执行以下命令，将这个拥有完整“V1 ➔ V2 演进树”的工业级项目提交：

```bash
git add .
git commit -m "docs: restructure README to showcase V1.0 baseline and V2.0 advanced RAG evolution"
git push origin main
```