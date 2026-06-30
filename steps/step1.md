### 🗺️ 项目全局总览 (System Architecture)

在动手之前，我们需要对我们要构建的 RAG（检索增强生成）系统有一个清晰的物理架构认知。

一个完整的 RAG 系统分为两条核心工作流：

```text
【离线文档处理流 (Data Ingestion Pipeline)】
原始文档 (PDF/TXT) ────► 1. 文档解析 (Loader) ────► 2. 文本切片 (Splitter) ────► 3. 向量化 (Embedding) ────► 4. 存入向量数据库 (Vector DB)

【在线问答交互流 (Query & Generation Pipeline)】
用户提问 (Query) ────► 1. 问题向量化 ────► 2. 检索相关片段 (Retriever) ────► 3. 拼接 Prompt (Context + Query) ────► 4. 大模型生成 (LLM) ────► 5. 吐出答案
```

在接下来的日子里，我们会把这个图里的每一个方框都亲手实现一遍。

---

### 🏁 第一步：环境搭建与 LangChain 极简调用

在第一步中，我们的目标是：**搭建纯净的开发环境，理解 LangChain 是如何抽象大模型接口的，并理解 LangChain 最核心的 LCEL（表达式语言）原理。**

#### 💡 核心原理剖析

##### 1. 为什么大模型开发需要虚拟环境？
在 Python 开发中，不同的项目依赖不同版本的库（比如 LangChain 升级频繁，旧项目和新项目容易冲突）。使用虚拟环境（如 `venv` 或 `conda`）可以为当前项目创建一个隔离的沙盒，避免污染全局环境。

##### 2. 什么是 ChatModel？它与传统 LLM 有什么区别？
在 LangChain 中，模型主要分为两类：
*   **LLMs：** 接受一个纯文本字符串输入，返回一个纯文本字符串输出（即 Text-In, Text-Out）。现在正逐渐被淘汰。
*   **ChatModels：** 专门为对话设计。它的输入不是简单的字符串，而是一个**消息列表 (List of Messages)**，每个消息有不同的角色（如 `SystemMessage` 系统设定、`HumanMessage` 用户输入、`AIMessage` 模型回复）。
*   *原理：* 现代大模型（如 GPT-4、Claude）底层虽然仍是预测下一个 Token，但在微调阶段（RLHF）都经过了对话格式的对齐。ChatModel 的结构更贴合现代 API。

##### 3. LangChain 的核心：LCEL (LangChain Expression Language) 是如何工作的？
在现代 LangChain 中，你会经常看到这样的代码：
`chain = prompt | model | parser`
这个竖线 `|` 是 Python 中的按位或运算符。LangChain 利用了 Python 的特殊方法 `__or__` 进行了重载。
*   *原理：* 每一个参与链式调用的组件都继承自 `Runnable` 类。`Runnable` 类实现了 `__or__` 方法。当你写 `A | B` 时，Python 底层实际上在执行 `A.__or__(B)`，它会返回一个新的 `RunnableSequence` 对象。
*   当执行 `chain.invoke(input)` 时，数据会像流水线一样：`input` 先传给 `prompt` 得到格式化后的消息，再传给 `model` 得到模型输出，最后传给 `parser` 提取出纯文本。

---

#### 🛠️ 动手实践

现在，请打开你的终端（Terminal）或命令行，跟着以下步骤操作：

##### 1. 创建项目文件夹并初始化虚拟环境
在你想存放项目的目录下执行：
```bash
# 创建项目文件夹
mkdir my-rag-assistant
cd my-rag-assistant

# 创建 Python 虚拟环境 (命名为 env)
python -m venv env

# 激活虚拟环境
# Windows 用户请执行:
env\Scripts\activate
# macOS/Linux 用户请执行:
source env/bin/activate
```
*激活成功后，你的终端提示符前应该会出现 `(env)` 字样。*

##### 2. 安装基础依赖
在激活的虚拟环境中，安装我们今天需要的库：
```bash
pip install langchain-core langchain-openai
```

##### 3. 编写你的第一个 LangChain 脚本
在项目文件夹下新建一个名为 `step1_simple_call.py` 的文件，写入以下代码。

*(注意：如果你没有 OpenAI 官方 Key，可以使用国内大模型如智谱、通义千问等，它们都提供了 OpenAI 兼容的 API 接口，只需要修改下面的 `base_url` 和 `api_key` 即可。)*

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 设置 API Key 和 接口地址 (请替换为你自己的 key 和 base_url)
# 如果使用的是国内代理或者其他大模型，请修改 base_url
os.environ["OPENAI_API_KEY"] = "你的_API_KEY"
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1" # 或者国内大模型的 API 终结点

# 2. 初始化 ChatModel
# 这里我们选用 gpt-4o-mini 或你所拥有额度的模型，设置 temperature 控制随机性（0 表示最严谨、最确定）
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 3. 创建 Prompt 模板
# PromptTemplate 用于将用户的动态输入，格式化为最终传递给大模型的结构化文本
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个资深的程序员助手。请用最通俗易懂、幽默的语言解答用户的问题。"),
    ("human", "{user_question}")
])

# 4. 创建输出解析器
# 大模型默认返回的是一个 AIMessage 对象，StrOutputParser 帮我们直接提取出里面的 content 文本
parser = StrOutputParser()

# 5. 使用 LCEL 组装链 (Chain)
# 这就是我们刚才提到的管道语法
chain = prompt | model | parser

# 6. 运行链
if __name__ == "__main__":
    question = "什么是 Python 中的装饰器？请用生活中的例子解释。"
    print(f"提问: {question}\n" + "-"*30)

    # invoke 方法是启动链的入口
    response = chain.invoke({"user_question": question})

    print("回答:\n", response)
```

##### 4. 运行脚本
在终端中运行：
```bash
python test_run.py
```

---

### 📝 你的第一步任务

请尝试完成上述步骤，并观察输出结果。

**完成或遇到问题后，请回复我，并告诉我：**
1. 你的代码是否运行成功？大模型给出了什么有趣的回答？
2. 如果报错了，请把报错信息贴给我，我们一起排查。
3. 如果理解了上面的 **ChatModel** 和 **LCEL 管道原理**，请告诉我。我们紧接着进入**第二步：文档解析与切片原理（Ingestion）**。


### 遇到的问题
#### 1. Q：为什么我在终端下载了langchain-core langchain-openai，但是用pycharm打开 test_run.py用from langchain_openai import ChatOpenAI显示没有包呢
A: 这是一个非常经典的“新手墙”，几乎每个刚接触 Python 开发的人都会遇到这个问题。

##### 💡 为什么会这样？（原理剖析）

这个现象背后的核心原理是：**Python 解释器（Interpreter）的多样性与隔离性**。

我们在第一步中通过 `python -m venv env` 创建了一个虚拟环境，并用 `pip` 把包下载到了这个名为 `env` 的沙盒里。
*   **终端的情况：** 你的终端激活了 `(env)`，使用的是 `env/bin/python`（或 `env\Scripts\python.exe`）这个解释器。所以你在终端能安装成功。
*   **PyCharm 的情况：** PyCharm 作为一个集成开发环境（IDE），默认并不知道你手动创建了哪个虚拟环境。它在打开这个项目时，可能默认关联了你电脑上的**全局 Python 解释器**（System Python），或者是它自己另外创建的一个新环境。因为全局 Python 里没有安装过 `langchain-openai`，所以它会报红线提示“找不到包”。

我们需要做的，就是**告诉 PyCharm 使用我们刚才创建的那个 `env` 虚拟环境作为项目的解释器**。

---

##### 🛠️ 解决步骤（在 PyCharm 中配置解释器）

请按照以下步骤在 PyCharm 中进行设置：

###### 第一步：打开设置窗口
*   **Windows / Linux 用户：** 菜单栏点击 `File` -> `Settings` (或者快捷键 `Ctrl + Alt + S`)。
*   **macOS 用户：** 菜单栏点击 `PyCharm` -> `Preferences` (或者快捷键 `Cmd + ,`)。

##### 第二步：找到 Python Interpreter（Python 解释器）
*   在左侧导航栏中，展开 `Project: my-rag-assistant`，点击 **`Python Interpreter`**。

##### 第三步：添加刚才创建的本地虚拟环境
1. 在右侧的 `Python Interpreter` 下拉框旁边，点击 **`Add Interpreter`**（添加解释器），选择 **`Add Local Interpreter...`**（添加本地解释器）。
2. 在弹出的窗口左侧选择 **`Virtualenv Environment`**。
3. 在右侧选择 **`Existing`**（现有环境，注意：不要选 New，我们要用刚才在终端建好的那个）。
4. 点击 **`Interpreter`** 输入框右侧的 `...` 文件夹图标，去浏览你电脑上的项目文件夹：
   * 找到你的项目目录 `my-rag-assistant` -> 进去找到 `env` 文件夹。
   * **Windows 用户：** 选择 `env \ Scripts \ python.exe`。
   * **macOS / Linux 用户：** 选择 `env / bin / python`。
5. 点击 **`OK`**，再点击 **`Apply`** (应用) 和 **`OK`** 保存退出。

---

##### 👀 验证是否成功
配置完成后，PyCharm 右下角会显示正在索引（Indexing...）。稍等十几秒，你会发现代码中的红线消失了。

此时，你可以直接在 PyCharm 中点击右上角的 **绿色三角形运行按钮**，或者右键 `test_run.py` 选择 **`Run 'test_run'`** 来运行代码。

请尝试配置一下，配置成功并运行出结果后告诉我，我们再继续下一步！如果依然报错，把报错信息发给我，我们随时调整。