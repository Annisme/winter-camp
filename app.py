"""Chainlit UI — PDF 問答系統"""

from typing import Literal
import shutil
import os
import chainlit as cl
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

from pdf_engine import sync_vectorstore, PDF_FOLDER

load_dotenv()


def build_agent(vectorstore):
    """建立 LangGraph agent，回傳 compiled app 和 retriever。"""
    retriever = vectorstore.as_retriever()

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
    model = ChatOpenAI(model="gpt-4.1", temperature=0).bind_tools(tools)

    def should_continue(state: MessagesState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app, retriever


@cl.on_chat_start
async def on_chat_start():
    """用戶開啟聊天時：同步 VectorStore、建立 agent。"""
    msg = cl.Message(content="正在同步 PDF 資料夾...")
    await msg.send()

    vectorstore = sync_vectorstore()
    app, retriever = build_agent(vectorstore)

    # 存到 session
    cl.user_session.set("app", app)
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("vectorstore", vectorstore)
    cl.user_session.set("thread_id", 42)

    msg.content = "PDF 同步完成！你可以開始提問，也可以上傳新的 PDF 檔案。"
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    """處理用戶訊息：上傳 PDF 或提問。"""

    # 檢查是否有上傳檔案
    if message.elements:
        uploaded_files = []
        for element in message.elements:
            if element.name and element.name.lower().endswith(".pdf"):
                os.makedirs(PDF_FOLDER, exist_ok=True)
                dest = os.path.join(PDF_FOLDER, element.name)
                shutil.copy(element.path, dest)
                uploaded_files.append(element.name)

        if uploaded_files:
            status_msg = cl.Message(content=f"正在處理上傳的檔案: {', '.join(uploaded_files)}...")
            await status_msg.send()

            vectorstore = sync_vectorstore()
            app, retriever = build_agent(vectorstore)
            cl.user_session.set("app", app)
            cl.user_session.set("retriever", retriever)
            cl.user_session.set("vectorstore", vectorstore)

            status_msg.content = f"已新增: {', '.join(uploaded_files)}，VectorStore 已更新！"
            await status_msg.update()

            # 如果只有上傳檔案沒有問題，就結束
            if not message.content or not message.content.strip():
                return

    # 處理提問
    app = cl.user_session.get("app")
    retriever = cl.user_session.get("retriever")
    thread_id = cl.user_session.get("thread_id")

    if not app:
        await cl.Message(content="系統尚未初始化，請稍候...").send()
        return

    # 呼叫 agent
    final_state = app.invoke(
        {"messages": [HumanMessage(content=message.content)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    answer = final_state["messages"][-1].content

    # 取得來源引用
    results = retriever.invoke(message.content)
    source_texts = []
    seen = set()
    for doc in results:
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            source_texts.append(
                cl.Text(
                    name=f"{source} 第{page}頁",
                    content=doc.page_content[:500],
                    display="side",
                )
            )

    # 發送回答 + 來源引用
    await cl.Message(content=answer, elements=source_texts).send()
