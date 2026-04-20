"""
LangGraph + Qwen3 聊天機器人範例
連接到本地 Qwen3-14B-FP8 模型，支援多輪對話與圖片輸入
"""

import base64
import os
import re
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
    max_tokens=8000,
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


# ── 5. 圖片處理工具 ────────────────────────────────────────────────────────────
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    """將圖片檔案編碼為 base64，回傳 (base64_data, media_type)"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"不支援的圖片格式：{ext}，支援格式：{', '.join(SUPPORTED_IMAGE_EXTENSIONS)}")
    
    media_type = IMAGE_MEDIA_TYPES[ext]
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, media_type


def is_image_url(text: str) -> bool:
    """判斷字串是否為圖片 URL"""
    url_pattern = re.compile(r'^https?://', re.IGNORECASE)
    if not url_pattern.match(text):
        return False
    ext = os.path.splitext(text.split("?")[0])[1].lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def build_human_message(text: str, image_sources: list[str]) -> HumanMessage:
    """
    建立包含文字與圖片的 HumanMessage。
    image_sources 可包含本地路徑或圖片 URL。
    """
    if not image_sources:
        return HumanMessage(content=text)

    # 多模態 content：list of dict
    content: list[dict] = []

    for source in image_sources:
        if is_image_url(source):
            # URL 圖片
            content.append({
                "type": "image_url",
                "image_url": {"url": source},
            })
        else:
            # 本地圖片 → base64
            try:
                b64_data, media_type = encode_image_to_base64(source)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64_data}"
                    },
                })
            except (ValueError, FileNotFoundError) as e:
                print(f"  ⚠️  圖片載入失敗（{source}）：{e}")

    # 文字放在圖片後面
    if text:
        content.append({"type": "text", "text": text})

    return HumanMessage(content=content)


def parse_image_input() -> list[str]:
    """
    提示使用者輸入圖片來源（本地路徑或 URL），支援多張。
    直接 Enter 略過。
    """
    print("  📎 圖片來源（本地路徑或 URL，多張請逐行輸入，留空結束）：")
    sources: list[str] = []
    while True:
        line = input("     > ").strip()
        if not line:
            break
        sources.append(line)
    return sources


# ── 6. 主程式：互動式聊天迴圈 ──────────────────────────────────────────────────
def main():
    app = build_graph()
    conversation_history: list[BaseMessage] = [
        SystemMessage(content="你是一個專業且簡潔的AI助手，請用繁體中文回答。")
    ]

    print("=" * 55)
    print("  Qwen3 聊天機器人（支援圖片輸入）")
    print("  輸入 'quit' / 'exit' 離開")
    print("  輸入 'image' 或 'img' 附加圖片後再輸入文字描述")
    print("=" * 55)

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

        # 判斷是否要附加圖片
        image_sources: list[str] = []
        if user_input.lower() in {"image", "img", "圖片"}:
            image_sources = parse_image_input()
            try:
                user_input = input("  💬 訊息（可留空）：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n再見！")
                break

        # 建立 HumanMessage（含圖片或純文字）
        human_msg = build_human_message(user_input, image_sources)
        conversation_history.append(human_msg)

        # 執行 Graph
        result = app.invoke({"messages": conversation_history})

        # 取得最新回覆
        ai_message: AIMessage = result["messages"][-1]
        print(f"\nQwen3：{ai_message.content}")

        # 更新對話歷史
        conversation_history = result["messages"]

    app.get_graph().draw_png("workflow.png")


if __name__ == "__main__":
    main()