"""Terminal 版 PDF 問答系統"""

from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from pdf_engine import sync_vectorstore

# Step 1: 從 .env 讀取 API Key
load_dotenv()

# Step 2: 同步 VectorStore
print("正在同步 PDF 資料夾...")
vectorstore = sync_vectorstore()
retriever = vectorstore.as_retriever()


# Step 3: 定義檢索工具
@tool
def retrieve_context(query: str):
    """Search for relevant documents from the PDFs. Returns content with source file and page number."""
    results = retriever.invoke(query)
    parts = []
    for doc in results:
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "?")
        parts.append(f"[來源: {source}, 第{page}頁]\n{doc.page_content}")
    return "\n\n".join(parts)


tools = [retrieve_context]
tool_node = ToolNode(tools)

# Step 4: LLM 模型
model = ChatOpenAI(model="gpt-4.1", temperature=0).bind_tools(tools)


def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Step 5: 建立 LangGraph workflow
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 儲存 Graph 結構圖
graph_png = app.get_graph(xray=True).draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(graph_png)
print("Graph 已儲存為 graph.png")

# Step 6: 互動式 terminal 提問
print("\n完成！輸入問題開始提問（輸入 quit 或 exit 退出）")
print("-" * 50)

thread_id = 42
while True:
    question = input("\n請輸入問題: ").strip()
    if question.lower() in ("quit", "exit"):
        print("再見！")
        break
    if not question:
        continue

    final_state = app.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    print("\n" + final_state["messages"][-1].content)
