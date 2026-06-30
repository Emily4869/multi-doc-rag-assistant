---

由于你已经完全吃透了切片的底层原理，我们直接进入最关键、最神奇的 **第三步：向量化与向量数据库（Vector Storage）**。

在这一步中，我们将把这些文本片段，变成计算机能够“听得懂”的数字，并把它们存进一个特殊的数据库中。

---

### 💡 第三步核心原理剖析

#### 1. 什么是 Embedding（向量化）？
大模型是看不懂中文文字的。为了让计算机理解文字的“意思”，科学家发明了 **Embedding（嵌入）** 技术。
*   **原理：** Embedding 模型（如 OpenAI 的 `text-embedding-3-small`）可以将任意长度的一段文本，转化为一个**固定长度的实数向量**（比如 1536 个浮点数组成的数组）。
*   **神奇特性：** 这个向量代表了这段文本的**语义空间坐标**。如果两段话的意思很相近（例如：“加班有饭补吗？” 和 “晚上八点后工作有餐饮补贴吗？”），它们转化出来的向量在多维空间中的**距离就会非常近**（通常用余弦相似度 Cosine Similarity 来计算）。

#### 2. 什么是向量数据库 (Vector Database)？
传统数据库（如 MySQL）是根据“关键字”进行精确匹配的。但用户提问往往千奇百怪。
*   如果用户搜：“加班有饭补吗？”
*   传统数据库去搜“饭补”，由于文档里只有“餐饮补贴”这个词，传统数据库就会返回“未找到”。
*   **向量数据库（如 Chroma、FAISS）**则是专门用来存储“向量”并进行**语义近似搜索**的。它不看字面是否一样，它只找“空间距离最近的向量”。它能识别出“饭补”和“餐饮补贴”在语义上是高度相关的，从而精准找到第二章。

---

### 🛠️ 动手实践

我们将安装向量数据库，并编写代码：**把上一阶段切好的三个片段存入向量数据库，并尝试进行一次“语义搜索”**。

#### 1. 安装向量数据库 Chroma
在终端中运行：
```bash
pip install chromadb
```
*(Chroma 是目前 AI 开发最常用的轻量级、开源向量数据库，支持本地运行，非常适合单机项目)*

#### 2. 编写向量存储与检索脚本
在项目文件夹下新建 `vector_store_demo.py`，写入以下代码。

*(注意：由于 Embedding 过程需要调用 OpenAI 的 Embedding 接口，请确保你的 `OPENAI_API_KEY` 已经正确配置。)*

```python
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 基础配置 (确保你的 key 和 base_url 正常)
os.environ["OPENAI_API_KEY"] = "你的_API_KEY"
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

def build_vector_store():
    # 2. 重新加载和切片文档（复用第二步的逻辑）
    loader = TextLoader("sample_company_policy.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = text_splitter.split_documents(documents)
    print(f"准备好 {len(chunks)} 个文本片段，开始向量化...")

    # 3. 初始化 Embedding 模型
    # 这里选用 OpenAI 推荐的最新、高性价比模型 text-embedding-3-small
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. 创建并保存向量数据库
    # persist_directory: 指定将数据库保存在本地哪个文件夹下（我们保存在本地的 ./chroma_db 目录）
    persist_db_path = "./chroma_db"
    
    print("正在计算向量并存入 Chroma 数据库...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_db_path
    )
    print(f"向量库创建成功！数据已保存在: {persist_db_path}")
    return vector_store

def search_test(vector_store):
    print("\n" + "="*20 + " 开始模拟语义检索 " + "="*20)
    
    # 模拟一个用户提问，注意：提问中【没有】出现文档中“餐饮补贴”或“锁屏”这些原词
    query = "晚上加班超过八点，公司给报销晚饭钱吗？"
    print(f"用户提问: '{query}'")
    
    # 使用 similarity_search 进行检索，k=1 表示只返回最相关的一条
    results = vector_store.similarity_search(query, k=1)
    
    if results:
        best_match = results[0]
        print(f"\n[检索成功] 找到最相关的文档片段：")
        print(f"内容:\n{best_match.page_content}")
        print(f"元数据: {best_match.metadata}")
    else:
        print("未找到相关片段。")

if __name__ == "__main__":
    # 执行构建
    db = build_vector_store()
    # 执行测试
    search_test(db)
```

#### 3. 运行代码
运行 `vector_store_demo.py`，观察控制台输出。
同时，留意你的项目文件夹下，是不是**多出了一个名为 `chroma_db` 的文件夹**？

---

### 📝 你的第三步任务

1. 运行代码后，向量数据库成功在本地创建了吗？检索的结果是否精准匹配到了关于“餐饮补贴”的第二章？
2. 仔细思考：用户问的是“晚饭钱”，而文档里写的是“餐饮补贴”。向量数据库为什么能把它们拼在一起？
3. 在运行成功后，观察你的项目目录。

完成这步后回复我，我们将进入 **第四步：利用 LangChain LCEL 组装 RAG 问答链（把检索内容和提问喂给大模型）**！

### 遇到的问题
#### Q1：我用的是glm模型，没有openai的api
A1：这个调整非常好！在实际的企业级项目开发中，**随时切换模型供应商（如从 OpenAI 切换到国产的智谱 GLM、通义千问，或本地部署的 Llama）是极常见的需求。**

这恰好能让你体会到 LangChain 最核心的优势之一：**统一的抽象接口**。

---

##### 💡 核心原理剖析：LangChain 的多供应商抽象（Unified Interface）

在传统的开发中，如果你从 OpenAI 换到智谱 GLM，你必须重写请求 API、解析 JSON 的全部底层代码。

但是在 LangChain 中，它提供了一个统一的 **`Embeddings` 基类**。
* 无论是 `OpenAIEmbeddings` 还是 `ZhipuAIEmbeddings`，它们都继承自同一个父类，并且都必须实现两个标准方法：
  1. `embed_documents(texts)`：把一堆文档切片转化成向量。
  2. `embed_query(text)`：把用户的一个问题转化成向量。
* 像向量数据库 Chroma 这样的第三方工具，它在设计时**只认这个标准接口，不认具体的品牌**。
* 因此，在下面的代码里，我们只需要把 Embedding 的初始化对象换成智谱的，后续所有的“切片、存储、检索”代码**一字不改**，系统就能完美运行。

---

##### 🛠️ 动手实践（智谱 GLM 版本）

我们需要使用智谱专用的 SDK 和 LangChain 对应的组件。

###### 1. 安装智谱 API 的依赖包
在你的虚拟环境中（确保终端有 `(env)`），安装智谱 SDK：
```bash
pip install zhipuai
```

###### 2. 修改 `vector_store_demo.py`
我们将代码中的 `OpenAIEmbeddings` 替换为 **`ZhipuAIEmbeddings`**，并指定智谱最新的向量模型 **`embedding-3`**。

修改后的完整代码如下：

```python
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入智谱专用的 Embedding 组件
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 配置智谱 API Key
# 智谱的 API Key 通常类似于 "xxxxxx.xxxxxx" 的格式
os.environ["ZHIPUAI_API_KEY"] = "你的_ZHIPU_API_KEY"

def build_vector_store():
    # 2. 加载和切片文档（与之前完全一致）
    loader = TextLoader("sample_company_policy.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = text_splitter.split_documents(documents)
    print(f"准备好 {len(chunks)} 个文本片段，开始向量化...")

    # 3. 初始化智谱 Embedding 模型
    # 我们使用智谱最新、性能最好的 embedding-3 向量模型
    embeddings = ZhipuAIEmbeddings(
        model="embedding-3"
    )

    # 4. 创建并保存向量数据库
    persist_db_path = "./chroma_db"
    
    print("正在通过智谱 API 计算向量并存入 Chroma 数据库...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_db_path
    )
    print(f"向量库创建成功！数据已保存在: {persist_db_path}")
    return vector_store

def search_test(vector_store):
    print("\n" + "="*20 + " 开始模拟语义检索 " + "="*20)
    
    # 模拟一个用户提问
    query = "晚上加班超过八点，公司给报销晚饭钱吗？"
    print(f"用户提问: '{query}'")
    
    # 执行检索，返回最相关的一条
    results = vector_store.similarity_search(query, k=1)
    
    if results:
        best_match = results[0]
        print(f"\n[检索成功] 找到最相关的文档片段：")
        print(f"内容:\n{best_match.page_content}")
        print(f"元数据: {best_match.metadata}")
    else:
        print("未找到相关片段。")

if __name__ == "__main__":
    # 执行构建并测试
    db = build_vector_store()
    search_test(db)
```

###### 3. 运行并测试
在 PyCharm 或终端中运行这个新脚本。

---

##### 📝 你的第三步（修改版）任务

1. 请填入你自己的智谱 API Key 并运行代码，观察检索结果。
2. 智谱的 `embedding-3` 是否成功帮我们匹配到了“第二章 财务报销流程（餐饮补贴）”的内容？
3. 通过这次修改，你感受到 LangChain 切换模型的便利性了吗？

运行成功后告诉我，我们将进入关键的 **第四步：组装完整的 RAG 问答链，让大模型（如 glm-4-flash）基于这些检索到的资料进行回答**！

#### Q2：输出为：准备好3个文本片段，开始向量化...正在计算向量并存入 Chroma 数据库...
A2：看到这个输出，说明代码成功启动了，并且文档已经切分完毕！

现在程序正停在 `正在计算向量并存入 Chroma 数据库...` 这一步，**这完全是正常的。**

##### 💡 此时后台正在发生什么？（底层原理）

在你看到这两行输出的时刻，LangChain 正在执行以下两件“重体力活”：

1. **网络网络请求（计算向量）：**
   `ZhipuAIEmbeddings` 正在把切分好的 3 个文本片段（共计约 400 个字）打包，通过 HTTPS 网络请求发送给智谱 AI 的服务器。
   智谱的服务器接收到文本后，利用他们的 `embedding-3` 大模型，将这些文本转化为三个高维实数向量（每个向量通常由上千个数字组成，代表这段文字的语义特征），然后将向量数据返回给你的电脑。
   *（这个过程依赖你的网络状况，通常需要 1~3 秒左右）*

2. **本地文件写入（保存数据库）：**
   收到向量数据后，Chroma 数据库开始工作。它会在你的项目目录下创建一个名为 `chroma_db` 的文件夹，并在其中创建 SQLite 数据库文件（保存文本和元数据）以及向量索引文件（保存刚才计算出来的浮点数向量）。

---

##### ❓ 确认你的运行状态

正常情况下，等待几秒钟后，控制台应该会继续吐出后面的日志：
```text
向量库创建成功！数据已保存在: ./chroma_db

==================== 开始模拟语义检索 ====================
用户提问: '晚上加班超过八点，公司给报销晚饭钱吗？'

[检索成功] 找到最相关的文档片段：
内容:
第四条 餐饮补贴：加班超过晚上20:00的员工，可享受每日40元的加班餐饮补贴。报销时需附上当晚加班系统截图与发票。
元数据: {'source': 'sample_company_policy.txt'}
```

请问你的程序**运行完了吗？是否成功输出了上述的“[检索成功]...”内容？** 

如果已经成功检索出“第四条 餐饮补贴”的内容，请告诉我。接下来，我们将迈入激动人心的 **第四步：利用大模型（GLM-4）进行“基于检索内容的智能问答”，完成真正的 RAG 闭环**！

#### Q3：显示进程已结束，退出代码为 -1073741819 (0xC0000005)
A3：这是一个非常关键的转折点！恭喜你，你遇到了大模型开发（尤其是 Windows 平台下）最经典、最折磨人的“硬核” Bug。

这个退出代码 `-1073741819 (0xC0000005)` 是 Windows 系统中的 **Access Violation（内存访问越界/段错误）**。

---

##### 💡 深度原理解析：为什么会闪退？

###### 1. 为什么 Python 没报错，直接闪退了？
在 Python 中，绝大多数错误（如我们刚才遇到的 `TypeError`）都是由 Python 虚拟机捕获并抛出的，你可以通过 `try...except` 抓住它们。
但是，当底层的 **C/C++ 动态链接库（DLL）** 发生物理内存越界时，它会绕过 Python 的异常捕获机制，操作系统会直接强行终止（Kill）整个 Python 进程。这就是为什么你没有看到任何 Python 报错，程序就突然退出了。

###### 2. 罪魁祸首是谁？
罪魁祸首是 **Chroma 数据库底层的 C++ 核心库：`hnswlib`**。
*   `hnsw`（分层导航小世界算法）是目前进行高维向量空间近似最近邻搜索最核心的算法。Chroma 为了追求极致的检索速度，底层是用 C++ 编写的 `hnswlib`。
*   在 Windows 10/11 的某些 Python 版本（尤其是 Python 3.11/3.12 配合特定 CPU 架构时），其预编译的二进制轮子（Wheel）与 Windows 的 C++ 运行时库或内存管理机制存在兼容性冲突，一调用就会直接触发系统的内存保护，导致进程崩溃。

---

##### 🛠️ 工程解决之道：无缝切换为 FAISS

作为一个合格的 AI 应用工程师，遇到此类环境底层 Bug 时，最明智的做法不是花几天时间去重编译 C++ 源码，而是**寻找更稳定、同样优秀的替代方案**。

我们要替换的库叫 **FAISS**（Facebook AI Similarity Search，由 Meta/脸书 开源）。
*   **地位：** FAISS 是业界最著名的、工业级的向量搜索加速库，支持在海量数据下进行超高速的语义检索，且其 Windows 版本的兼容性极佳。
*   **体现 LangChain 的威力：** 因为 LangChain 实现了完美的“解耦”，我们只需修改 **2 行代码**，把数据库从 `Chroma` 换成 `FAISS`，其他所有代码和之前的逻辑完全不用动！

###### 步骤一：安装 FAISS 依赖
在你的虚拟环境中安装 `faiss-cpu`（不需要显卡，CPU 版本即可）：
```bash
pip install faiss-cpu
```

###### 步骤二：修改 `vector_store_demo.py` 为 FAISS 版本
请将你的代码修改为以下内容。请仔细对照我们是怎么用 `FAISS` 替换 `Chroma` 的：

```python
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
# 1. 替换导入：将 Chroma 替换为 FAISS
from langchain_community.vectorstores import FAISS

# 配置智谱 API Key
os.environ["ZHIPUAI_API_KEY"] = "你的_ZHIPU_API_KEY"

def build_vector_store():
    # 加载和切片文档
    loader = TextLoader("sample_company_policy.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = text_splitter.split_documents(documents)
    print(f"准备好 {len(chunks)} 个文本片段，开始向量化...")

    # 初始化智谱 Embedding 模型
    embeddings = ZhipuAIEmbeddings(model="embedding-3")

    # 2. 替换创建方法：使用 FAISS.from_documents
    print("正在通过智谱 API 计算向量并存入 FAISS...")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # 3. 替换保存方法：FAISS 使用 save_local 保存到本地文件夹
    persist_db_path = "./faiss_index"
    vector_store.save_local(persist_db_path)
    print(f"向量库创建成功！数据已保存在本地文件夹: {persist_db_path}")
    return vector_store

def search_test(vector_store):
    print("\n" + "="*20 + " 开始模拟语义检索 " + "="*20)
    
    # 模拟一个用户提问
    query = "晚上加班超过八点，公司给报销晚饭钱吗？"
    print(f"用户提问: '{query}'")
    
    # FAISS 的 similarity_search 用法与 Chroma 完全一致！
    results = vector_store.similarity_search(query, k=1)
    
    if results:
        best_match = results[0]
        print(f"\n[检索成功] 找到最相关的文档片段：")
        print(f"内容:\n{best_match.page_content}")
        print(f"元数据: {best_match.metadata}")
    else:
        print("未找到相关片段。")

if __name__ == "__main__":
    # 执行构建并测试
    db = build_vector_store()
    search_test(db)
```

---

##### 📝 你的新任务

1. 运行修改后的 `vector_store_demo.py`。
2. 观察控制台，看看是否还会闪退？程序是否能顺利输出检索成功的结果？
3. 检查你的项目目录，是否生成了一个名为 `faiss_index` 的新文件夹？

这个坑在面试中是非常好的谈资，展现了你对系统底层（C++、内存保护、库解耦）和实际排产经验的深刻理解。

运行成功后告诉我，我们立即前往 **第四步：构建 RAG 完整问答链（LLM Generation）**！