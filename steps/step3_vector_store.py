import  os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 设置 API Key 和 接口地址
os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPU_API_KEY")
# os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"

def build_vector_store():
    # 2. 重新加载和切片文档（复用第二步的逻辑）
    loader = TextLoader("sample_company_policy.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = text_splitter.split_documents(documents)
    print(f"准备好{len(chunks)}个文本片段，开始向量化...")

    # 3. 初始化 Embedding 模型
    # 这里选用 OpenAI 推荐的最新、高性价比模型 text-embedding-3-small
    # 3. 初始化智谱 Embedding 模型
    # 我们使用智谱最新、性能最好的 embedding-3 向量模型
    embeddings = ZhipuAIEmbeddings(model="embedding-3")

    # # 4. 创建并保存向量数据库
    # # persist_directory: 指定将数据库保存在本地哪个文件夹下（我们保存在本地的 ./chroma_db 目录）
    # persist_db_path = "./chroma_db"
    #
    # print("正在计算向量并存入 Chroma 数据库...")
    # vector_store = Chroma.from_documents(
    #     documents=chunks,
    #     embedding=embeddings,
    #     persist_directory=persist_db_path
    # )
    # print(f"向量库创建成功！数据已保存在：{persist_db_path}")
    # return vector_store


    # 2. 替换创建方法：使用 FAISS.from_documents
    print("正在通过智谱 API 计算向量并存入 FAISS...")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    # 3. 替换保存方法：FAISS 使用 save_local 保存到本地文件夹
    persist_db_path = "../faiss_index"
    vector_store.save_local(persist_db_path)
    print(f"向量库创建成功！数据已保存在本地文件夹: {persist_db_path}")
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