import os
from langchain_community.chat_models import  ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 配置智谱 API Key
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPU_API_KEY")

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
    # 2. 线上加载本地已有的 FAISS 向量数据库
    print("正在加载本地向量数据库...")
    embeddings = ZhipuAIEmbeddings(model="embedding-3")

    # 注意：allow_dangerous_deserialization 必须设置为 True，因为这是我们自己生成的安全文件
    vector_store = FAISS.load_local(
        folder_path ="../faiss_index",
        embeddings = embeddings,
        allow_dangerous_deserialization= True
    )

    # 3. 将向量库转化为“检索器（Retriever）”
    # search_kwargs={"k": 1} 表示我们只检索最相关的一条片段，作为上下文
    # 💥 工程优化：将检索数量 k 从 1 提升到 3
    # 这样可以大幅容忍向量模型打分的微小误差，保证正确片段能喂给大模型
    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    # 4. 初始化智谱聊天模型
    # 这里我们选用性价比较高、响应快速的 glm-4-flash 模型
    llm = ChatZhipuAI(
        model="glm-4-flash",
        temperature = 0   # 设为 0，代表我们需要极其严谨、没有创造力的回答
    )

    # 5. 设计 RAG 专用的系统提示词（约束大模型的回答范围）
    system_prompt = (
        "你是一个专业且温馨的企业知识库助手。\n"
        "请结合下方提供的【背景资料】，回答用户的问题。\n\n"
        "【答题原则】：\n"
        "1. 允许并鼓励进行符合生活常识的合理推理（例如：晚上7点属于18:00之后；'晚饭钱'、'加班餐'可等同于'餐饮补贴'）。\n"
        "2. 回答时请直接给出结论和具体规定（如金额、流程等），并保持语气自然、专业。\n"
        "3. 如果【背景资料】中确实没有提及与问题相关的任何线索，再回答：'抱歉，根据公司规定守则，我没有找到相关信息。'\n\n"
        "【背景资料】:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            ("human","{question}")
        ]
    )

    # 6. 利用 LCEL 组装完整的 RAG 链
    # 运行机制解密：
    # 1. 用户的提问作为一个字符串传入。
    # 2. RunnablePassthrough() 负责把用户的原问题原封不动地传给 "question" 字段。
    # 3. retriever | format_docs 负责把同一个提问传给检索器，搜到片段后用 format_docs 拼成字符串，传给 "context" 字段。
    # 4. 拼装好的 {"context": ..., "question": ...} 字典被送入 prompt 实例化。
    # 5. 格式化后的 Prompt 消息传给 llm 生成回答。
    # 6. 最后通过 StrOutputParser 提取文本。
    rag_chain = (
        {"context":retriever | format_docs, "question":RunnablePassthrough()}
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
    print("=" * 40)


if __name__ == "__main__":
    run_rag_assistant()