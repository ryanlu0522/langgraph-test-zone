"""
LangGraph + Qwen3 聊天機器人範例
連接到本地 Qwen3-14B-FP8 模型，支援多輪對話
"""

from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ── 1. 定義狀態 ────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. 初始化模型（OpenAI 相容介面）────────────────────────────────────────────
llm = ChatOpenAI(
    model="Qwen/Qwen3.5-4B",
    base_url="http://10.39.72.60:8000/v1",
    api_key="dummy",           # vLLM 不需要真實 API key
    temperature=0.7,
    max_tokens=1024,
)


# ── 3. 定義節點：呼叫 LLM ──────────────────────────────────────────────────────
def chatbot_node(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ── 4. 建立 LangGraph ──────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)
    return graph.compile()


# ── 5. 主程式：互動式聊天迴圈 ──────────────────────────────────────────────────
def main():
    app = build_graph()
    #conversation_history: list[BaseMessage] = []
    conversation_history: list[BaseMessage] = [
        SystemMessage(content="你是一個專業且簡潔的AI助手，請用繁體中文回答。")
    ]

    print("=" * 50)
    print("  Qwen3 聊天機器人（輸入 'quit' 或 'exit' 離開）")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n再見！")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye", "再見"}:
            print("再見！")
            break

        # 加入使用者訊息
        conversation_history.append(HumanMessage(content=user_input))

        # 執行 Graph（帶入完整對話歷史）
        result = app.invoke({"messages": conversation_history})

        # 取得最新回覆
        ai_message: AIMessage = result["messages"][-1]
        print(f"\nQwen3：{ai_message.content}")

        # 更新對話歷史（使用 graph 回傳的完整 messages）
        conversation_history = result["messages"]

    app.get_graph().draw_png("workflow.png")


if __name__ == "__main__":
    main()