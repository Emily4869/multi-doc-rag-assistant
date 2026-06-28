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