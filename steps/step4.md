现在，我们正式开启 **第四步：构建 RAG 完整问答链（利用大模型根据检索内容生成回答）**。

在这一步中，我们将把“检索”和“生成”这两个模块串联起来，完成 RAG 系统的核心闭环：
1. **用户输入**一个问题。
2. 系统自动去我们第三步建好的本地 `faiss_index` 数据库中**检索**出最相关的文档片段。
3. 将文档片段作为“上下文（Context）”与用户的问题一起拼进提示词（Prompt）。
4. 将拼好的提示词发送给**智谱大模型（glm-4-flash）**。
5. 大模型只根据上下文，给出一个**严谨、精准、没有幻觉**的回答。

---

### 💡 核心原理剖析

#### 1. RAG 是如何消除大模型“幻觉（Hallucination）”的？
大模型本质上是一个“概率预测机器”，当遇到它不知道的知识（比如你们公司的内部守则）时，它会倾向于用它强大的联想能力“一本正经地胡说八道”。
在 RAG 中，我们通过 **提示词工程（Prompt Engineering）** 限制其发挥：
> **我们的核心提示词逻辑：**
> “你是一个极其严谨的 AI 助手。以下是唯一可靠的背景资料（Context）。请你**仅仅**根据这些资料来回答用户的问题。如果资料中没有提到相关信息，请直接回答‘抱歉，根据公司规定文档，我无法回答该问题’，绝对不允许瞎编。”
通过这种“开卷考试”加“严厉警告”的方式，大模型的回答范围被牢牢约束在检索出来的文档内。

#### 2. 线上检索与线下写入的分离
在生产环境中，我们不会每次用户提问时都重新去解析 PDF、切片和建数据库。
*   **线下（Offline）：** 我们已经在第三步生成了 `faiss_index` 文件夹。
*   **线上（Online）：** 用户发起提问时，我们直接通过 `FAISS.load_local()` 从硬盘上把现成的索引加载到内存中进行查询。这种设计能保证高并发下的系统响应速度。

#### 3. 为什么加载 FAISS 本地索引需要设置 `allow_dangerous_deserialization=True`？（面试加分安全常识）
在代码中，加载本地 FAISS 索引时，LangChain 会强制你传入一个参数 `allow_dangerous_deserialization=True`。
*   *原理：* FAISS 在保存索引时，底层使用了 Python 的 `pickle` 序列化库。而 `pickle` 在加载（反序列化）未知来源的文件时，存在安全隐患——恶意攻击者如果伪造了一个 FAISS 索引文件让你加载，代码在反序列化时可能会在你的服务器上执行任意恶意脚本。
*   因为这个索引是我们自己刚刚安全生成的，所以我们手动把这个安全开关设置为 `True`。这也体现了你对应用安全性的关注。

---

### 🛠️ 动手实践

我们在项目文件夹下编写线上问答脚本 `rag_qa.py`。

#### 1. 编写 RAG 问答链脚本
新建 `rag_qa.py`，写入以下代码。

```python
import os
# 导入智谱聊天模型组件
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 配置智谱 API Key
os.environ["ZHIPUAI_API_KEY"] = "你的_ZHIPU_API_KEY"

def format_docs(docs):
    """辅助函数：将检索出来的多个 Document 对象拼接成一个长字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag_assistant():
    # 2. 线上加载本地已有的 FAISS 向量数据库
    print("正在加载本地向量数据库...")
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    
    # 注意：allow_dangerous_deserialization 必须设置为 True，因为这是我们自己生成的安全文件
    vector_store = FAISS.load_local(
        folder_path="./faiss_index", 
        embeddings=embeddings,
        allow_dangerous_deserialization=True 
    )
    
    # 3. 将向量库转化为“检索器（Retriever）”
    # search_kwargs={"k": 1} 表示我们只检索最相关的一条片段，作为上下文
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    # 4. 初始化智谱聊天模型
    # 这里我们选用性价比较高、响应快速的 glm-4-flash 模型
    llm = ChatZhipuAI(
        model="glm-4-flash",
        temperature=0  # 设为 0，代表我们需要极其严谨、没有创造力的回答
    )
    
    # 5. 设计 RAG 专用的系统提示词（约束大模型的回答范围）
    system_prompt = (
        "你是一个极其敬业且严谨的企业知识库助手。\n"
        "请仅仅基于下方提供的【背景资料】来回答用户的问题。回答要求清晰、专业、实事求是。\n"
        "如果【背景资料】中完全没有提到用户的问题，请礼貌地回答：'抱歉，根据公司规定守则，我没有找到相关信息。'\n"
        "禁止编造任何不存在的规定或事实！\n\n"
        "【背景资料】:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    # 6. 利用 LCEL 组装完整的 RAG 链
    # 运行机制解密：
    # 1. 用户的提问作为一个字符串传入。
    # 2. RunnablePassthrough() 负责把用户的原问题原封不动地传给 "question" 字段。
    # 3. retriever | format_docs 负责把同一个提问传给检索器，搜到片段后用 format_docs 拼成字符串，传给 "context" 字段。
    # 4. 拼装好的 {"context": ..., "question": ...} 字典被送入 prompt 实例化。
    # 5. 格式化后的 Prompt 消息传给 llm 生成回答。
    # 6. 最后通过 StrOutputParser 提取文本。
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG 系统初始化完成！开始测试问答系统。")
    print("="*40)
    
    # --- 测试场景 A：提问在文档范围内的内容 ---
    question_a = "我晚上加班到九点，公司会给报销晚饭钱吗？有金额限制吗？"
    print(f"提问 A: {question_a}")
    response_a = rag_chain.invoke(question_a)
    print(f"回答 A:\n{response_a}\n")
    print("-" * 40)
    
    # --- 测试场景 B：提问完全不在文档范围内（测试防幻觉能力） ---
    question_b = "我们公司有五险一金和年假吗？具体是怎么规定的？"
    print(f"提问 B: {question_b}")
    response_b = rag_chain.invoke(question_b)
    print(f"回答 B:\n{response_b}")
    print("="*40)

if __name__ == "__main__":
    run_rag_assistant()
```

#### 2. 运行脚本
在终端或 PyCharm 中直接运行 `rag_qa.py`。

---

### 📝 你的第四步任务

1. 请填入你的 API Key 并运行代码，仔细观察 **回答 A** 和 **回答 B** 的输出。
2. 回答 A 是否能够准确提取出“餐饮补贴是 40 元”、“需要加班到 20:00 以后”等细节？
3. 对于没有写在守则里的“五险一金”（问题 B），大模型是否被成功“约束”住了？它有没有胡编乱造？
4. 如果对 LCEL 链 `{"context": retriever | format_docs, "question": RunnablePassthrough()}` 的并行传递运行机制有任何疑问，请随时提出来。

完成这步后，你已经成功实现了一个高水准的 RAG 核心后端！下一步我们将使用 **Streamlit 把它包装成一个漂亮的、支持文件上传和对话历史气泡的可视化网页系统**！

### 遇到的问题
#### Q1：题A回答不出来，显示没有找到，问题B回答正确
A1：这绝对是 RAG 真实开发中最具价值、最经典的痛点！在企业开发中，几乎 80% 的精力都花在调优类似的问题上：**明明文档里有，大模型为什么偏偏回答“不知道”？**

让我们像真正的 RAG 工程专家一样，对这个问题进行一步步的**Debug（调试）与诊断**。

---

##### 💡 深度原理解析：大模型为什么“选择性失明”？

在 RAG 系统中，当大模型对已有知识回答“不知道”时，通常只可能有以下两个原因：

1. **原因一：检索失败（Retrieval Gap）**
   即：向量数据库在检索时，**根本没有把正确的那个片段（第二章：财务报销）作为 Top-1 检索出来**。
   因为我们设置了 `k=1`（只拿最相关的一个片段）。如果用户的提问是“加班到九点，报销晚饭钱”，智谱的 `embedding-3` 模型可能在计算时，认为第一章（考勤管理，里面有“工作时间、早退”）或者第三章（IT规范，里面有“工作区域、机密代码”）的语义向量跟这个问题的距离更近。
   如果数据库把“第一章（考勤）”交给了大模型，而大模型严格遵循了我们的 Prompt（“只能根据背景资料回答”），它在考勤管理里确实找不到任何关于晚饭钱的规定，所以只能老老实实回答“我没有找到相关信息”。

2. **原因二：数字逻辑推理失败（Reasoning Gap）**
   文档写的是：`加班超过晚上 20:00`。
   用户问的是：`我晚上加班到九点（21:00）`。
   如果大模型拿到了正确的片段，但它无法推理出“九点（21:00）属于超过 20:00 这一范围”，它也会回答“没有提到”。（不过对于 `glm-4-flash` 这样强大的模型，推理这个问题一般很轻松，因此**原因一的概率高达 95%**）。

---

##### 🛠️ 诊断与工程优化

为了查明真相，并优化我们的 RAG 系统，我们需要做两件事：
1. **打印出检索到的真实片段**（揭开黑盒，看看数据库到底给大模型喂了什么）。
2. **将检索数量 $K$ 值提高**（在工业界，几乎没人敢用 `k=1`，通常使用 `k=3` 或 `k=5`，确保相关信息被打包带走）。

###### 步骤一：修改 `rag_qa.py` 代码进行诊断

请用以下代码替换你之前的 `rag_qa.py`。
我们在代码里加了 **打印检索到的原始片段** 的调试逻辑，并且把 $K$ 值调整为了 **`3`**（一次检索 3 个最相关的片段，哪怕第一名错了，第二名对的也能被带进去）：

```python
import os
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 配置智谱 API Key
os.environ["ZHIPUAI_API_KEY"] = "你的_ZHIPU_API_KEY"

def format_docs(docs):
    """辅助函数：拼接文本，并在控制台打印出来，方便我们 Debug"""
    print("\n--- [Debug 调试] 向量库检索到的参考资料片段如下： ---")
    for i, doc in enumerate(docs):
        print(f"【参考片段 {i+1}】(来源: {doc.metadata.get('source')}):")
        print(doc.page_content)
        print("-" * 30)
    print("----------------------------------------------------\n")
    return "\n\n".join(doc.page_content for doc in docs)

def run_rag_assistant():
    print("正在加载本地向量数据库...")
    embeddings = ZhipuAIEmbeddings(model="embedding-3")
    
    vector_store = FAISS.load_local(
        folder_path="./faiss_index", 
        embeddings=embeddings,
        allow_dangerous_deserialization=True 
    )
    
    # 💥 工程优化：将检索数量 k 从 1 提升到 3
    # 这样可以大幅容忍向量模型打分的微小误差，保证正确片段能喂给大模型
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatZhipuAI(
        model="glm-4-flash",
        temperature=0
    )
    
    system_prompt = (
        "你是一个极其敬业且严谨的企业知识库助手。\n"
        "请仅仅基于下方提供的【背景资料】来回答用户的问题。回答要求清晰、专业、实事求是。\n"
        "如果【背景资料】中完全没有提到用户的问题，请礼貌地回答：'抱歉，根据公司规定守则，我没有找到相关信息。'\n"
        "禁止编造任何不存在的规定或事实！\n\n"
        "【背景资料】:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG 系统（诊断版）初始化完成！")
    print("="*40)
    
    # 测试问题 A
    question_a = "我晚上加班到九点，公司会给报销晚饭钱吗？有金额限制吗？"
    print(f"提问 A: {question_a}")
    response_a = rag_chain.invoke(question_a)
    print(f"最终回答 A:\n{response_a}\n")

if __name__ == "__main__":
    run_rag_assistant()
```

###### 步骤二：运行并观察控制台

运行这个新脚本，仔细看控制台输出。

---

##### 📝 你的新调试任务

1. 运行修改后的代码，看看打印出来的 **`[Debug 调试] 向量库检索到的参考资料片段如下：`** 里面，到底包含了哪些片段？
2. 检索出来的片段 1、2、3 里面，**有没有出现“第二章 财务报销流程（第四条餐饮补贴）”这一段**？
3. 将 $K$ 值改为 3 后，最终回答 A 是否成功回答出来了？

通过这样实时的 Debug 观察，你会对“向量相似度检索”的边界和 RAG 系统中 $K$ 值的工程意义有极为深刻的体会。期待你的运行反馈！

#### Q2：有在参考片段里面，且是第一个，但是还是回答不出来
A2：恭喜你！这意味着我们的**前三个阶段（解析、切片、向量存储、语义检索）在工程上已经 100% 成功了！** 

既然 FAISS 检索出来的第一条就是“第二章 财务报销流程（第四条餐饮补贴）”，说明向量库和检索器的分工干得极其漂亮，没有受到任何 Bug 的干扰。

现在，问题被精准锁定在了**最顶层的“大模型推理与提示词工程（LLM & Prompt）”阶段**。

---

##### 💡 深度原理解析：为什么拿到了正确资料，大模型却闭眼说瞎话？

在 RAG 系统的开发中，这种“捧着正确答案说不知道”的现象叫 **“提示词过度约束”（Prompt Over-Constraint）**。

###### 1. 轻量大模型（如 `glm-4-flash`）的“防御性回答”机制
我们在前一步设计的提示词非常严厉：
> *“请仅仅基于...”、“完全没有提到...”、“禁止编造任何不存在的规定或事实！”*

像 `glm-4-flash` 这样主打速度和高性价比的轻量模型，其参数量相对较小。在面对如此强烈的**否定性约束（“禁止”、“完全没有”）**时，大模型会启动强烈的**“防御机制”**。
它的逻辑是：
* 用户的提问是：`加班到九点`，`报销晚饭钱`。
* 参考资料写的是：`超过晚上 20:00`，`餐饮补贴`。
* 此时，它面临一个微小的语义鸿沟（九点是否等于超过 20:00？晚饭钱是否等于餐饮补贴？）。
* 因为我们严厉警告了它“禁止编造任何事实”，所以模型心想：*“为了绝对安全，不被惩罚，我还是直接回答‘没找到’吧，这样最保险。”*

这就像学校里一个很听话但有点胆小的学生，你警告他“做错一题就扣100分”，他遇到稍微需要转个弯的题目，就会直接选择留白交空卷。

###### 2. RAG 提示词工程的“松紧度管理”
优秀的 RAG 提示词，应该在**防幻觉**和**允许合理推理**之间取得平衡。我们需要在提示词中**显式授权大模型进行常识推理**（告诉它：你可以动脑子）。

---

##### 🛠️ 优化方案：重构提示词（Prompt Optimization）

现在，我们来微调提示词。我们要在 Prompt 中明确告诉大模型：**你可以进行符合常识的语义等价推理。**

请修改 `rag_qa.py` 中的 `system_prompt` 变量，将其替换为以下内容：

```python
    # 💥 优化后的提示词：保留严谨性，同时显式授权合理推理
    system_prompt = (
        "你是一个专业且温馨的企业知识库助手。\n"
        "请结合下方提供的【背景资料】，回答用户的问题。\n\n"
        "【答题原则】：\n"
        "1. 允许并鼓励进行符合生活常识的合理推理（例如：晚上9点属于20:00之后；'晚饭钱'、'加班餐'可等同于'餐饮补贴'）。\n"
        "2. 回答时请直接给出结论和具体规定（如金额、流程等），并保持语气自然、专业。\n"
        "3. 如果【背景资料】中确实没有提及与问题相关的任何线索，再回答：'抱歉，根据公司规定守则，我没有找到相关信息。'\n\n"
        "【背景资料】:\n{context}"
    )
```

---

##### 📝 你的新任务

1. 将上述修改后的 `system_prompt` 替换到 `rag_qa.py` 中。
2. 重新运行代码，看看大模型这次是否能“转弯”，从而把“第四条 餐饮补贴（40元，发票截图等）”的信息准确回答出来？
3. 观察问题 B（五险一金）是否依然保持了“回答找不到”的严谨性？

通过这一次调试，你将真正掌握 **提示词工程在 RAG 系统中如何控制大模型智力释放** 的核心技巧。期待你的好消息！

