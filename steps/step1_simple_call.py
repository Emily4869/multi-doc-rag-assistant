import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 设置 API Key 和 接口地址
os.environ["OPENAI_API_KEY"] = os.getenv("ZHIPU_API_KEY")
# os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4"

# 2. 初始化 ChatModel
model = ChatOpenAI(
    model = "glm-4",
    temperature=0
)

# 3. 创建 Prompt 模板
# PromptTemplate 用于将用户的动态输入，格式化为最终传递给大模型的结构化文本
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", "你是一个资深的程序员助手。请用最通俗易懂、幽默的语言解答用户的问题。"
        ),
        (
            "human", "{user_question}"
        )
    ]
)

# 4. 创建输出解析器
# 大模型默认返回的是一个 AIMessage 对象，StrOutputParser 帮我们直接提取出里面的 content 文本
parser = StrOutputParser()

# 5. 使用 LCEL 组装链 (Chain)
# 这就是我们刚才提到的管道语法
chain = prompt | model | parser

# 6. 运行链
if __name__ == "__main__":
    question = "什么是 Python 中的装饰器？请用生活中的例子解释。"
    print(f"提问: {question}\n" + "-" * 30)

    # invoke 方法是启动链的入口
    response = chain.invoke({"user_question": question})

    print("回答:\n", response)