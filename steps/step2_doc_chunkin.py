from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document(file_path):
    # 1. 初始化加载器 (Loader)
    # TextLoader 用于读取纯文本文件。如果你用的是 PDF，可以使用 PyPDFLoader
    print(f"正在加载文档：{file_path}")
    loader = TextLoader(file_path, encoding = "utf-8")
    documents = loader.load()

    # 2. 初始化切片器 (Text Splitter)
    # chunk_size: 每个片段的最大字符数
    # chunk_overlap: 相邻片段之间的重叠字符数
    """
    RecursiveCharacterTextSplitter 的工作流程是这样的：
它首先尝试用优先级最高的换行符 \n\n（即空行）去切分整篇文档。
切完之后，它发现整篇文档被分成了三大块（第一章、第二章、第三章）。
接着，它测量了这三大块的字符长度。正如你所发现的，每一章的字符数大约在 100~135 之间。
此时，切片器看了一下你设置的 chunk_size=150。它心想：“既然这三块的长度都已经小于 150 了，那我就不需要再对它们进行更细的二次切分了。”
既然不需要二次切分，那么自然就不需要产生重叠区（Overlap）。重叠区只在“某一块文本超长、被迫切成两半”时，为了连接语义才会产生。
    """


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,       # 故意设小一点，方便我们观察切片效果
        chunk_overlap=30,      # 重叠区，既然不需要二次切分，那么自然就不需要产生重叠区（Overlap）。重叠区只在“某一块文本超长、被迫切成两半”时，为了连接语义才会产生。
        length_function=len,
    )

    # 3. 执行切片
    chunks = text_splitter.split_documents(documents)

    # 4. 打印切片结果以便分析
    print(f"\n文档加载成功！原始文档数: {len(documents)}")
    print(f"切片完成！共切分成 {len(chunks)} 个片段。")
    print("-" * 50)

    for i, chunk in enumerate(chunks):
        print(f"【片段{i+1}】（字符数：{len(chunk.page_content)}）")
        print(chunk.page_content)
        # 打印元数据，告诉你这个片段来自哪个文件的第几页/行
        print(f"元数据: {chunk.metadata}")
        print("-" * 50)
if __name__ == "__main__":
    process_document("../sample_company_policy.txt")