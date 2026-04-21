"""
LangGraph + Qwen3 聊天機器人（含 Tool Use）
雙重偵測：優先 LangChain tool_calls，fallback 到 <tool_call> regex 解析
"""

import base64
import json
import os
import re
from datetime import datetime
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ── 1. 狀態 ────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── 2. 工具定義 ────────────────────────────────────────────────────────────────
@tool
def get_current_date() -> str:
    """取得今天的日期，包含年、月、日與星期資訊。"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return (
        f"今天是 {now.year} 年 {now.month} 月 {now.day} 日，{weekdays[now.weekday()]}。"
        f"（完整日期時間：{now.strftime('%Y-%m-%d %H:%M:%S')}）"
    )


@tool
def get_current_time() -> str:
    """取得現在的時間（時、分、秒）。"""
    return f"現在時間是 {datetime.now().strftime('%H:%M:%S')}。"


TOOLS = [get_current_date, get_current_time]
TOOL_MAP = {t.name: t for t in TOOLS}


# ── 3. LLM 初始化 ──────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="Qwen/Qwen3.5-9B",
    base_url="http://10.39.72.60:8000/v1",
    api_key="dummy",
    temperature=0.7,
    max_tokens=8000,
)
llm_with_tools = llm.bind_tools(TOOLS)


# ── 4. Fallback：解析 Qwen3 原生 <tool_call> 格式 ─────────────────────────────
# 支援兩種常見格式：
#   <tool_call>\n<function=NAME>\n{json}\n</function>\n</tool_call>
#   <tool_call>\n{"name": "NAME", "arguments": {...}}\n</tool_call>
_FUNC_TAG_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_JSON_TAG_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_raw_tool_calls(text: str) -> list[dict]:
    """
    從模型原始文字中解析工具呼叫，回傳 list of {'name': ..., 'args': ...}
    """
    results = []

    # 格式一：<function=NAME>
    for m in _FUNC_TAG_RE.finditer(text):
        name = m.group(1).strip()
        raw_args = m.group(2).strip()
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}
        results.append({"name": name, "args": args})

    if results:
        return results

    # 格式二：{"name": "...", "arguments": {...}}
    for m in _JSON_TAG_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            name = obj.get("name") or obj.get("function") or obj.get("tool")
            args = obj.get("arguments") or obj.get("args") or {}
            if name:
                results.append({"name": name, "args": args})
        except json.JSONDecodeError:
            pass

    return results


def clean_tool_call_tags(text: str) -> str:
    """移除輸出中殘留的 <tool_call> 區塊。"""
    text = _FUNC_TAG_RE.sub("", text)
    text = _JSON_TAG_RE.sub("", text)
    return text.strip()


# ── 5. 節點 ────────────────────────────────────────────────────────────────────
def chatbot_node(state: State) -> State:
    print('======================= chatbot_node work')
    response: AIMessage = llm_with_tools.invoke(state["messages"])
    #print(f'<<<<<<< {response} >>>>>>>')
    return {"messages": [response]}


def tool_node(state: State) -> State:
    print('======================= tool_node work')
    last: AIMessage = state["messages"][-1]
    print('==============> ', last)
    tool_results = []

    # 優先使用 LangChain 解析好的 tool_calls
    calls = last.tool_calls if (hasattr(last, "tool_calls") and last.tool_calls) else []

    # Fallback：自行解析原始文字
    if not calls:
        raw_calls = parse_raw_tool_calls(last.content)
        calls = [
            {"name": c["name"], "args": c["args"], "id": f"fallback_{i}"}
            for i, c in enumerate(raw_calls)
        ]

    for call in calls:
        name = call["name"]
        print(f"\n  🔧 呼叫工具：{name}")
        if name in TOOL_MAP:
            result = TOOL_MAP[name].invoke(call.get("args", {}))
        else:
            result = f"❌ 未知工具：{name}"
        print(f"  📋 結果：{result}")
        tool_results.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return {"messages": tool_results}


def should_use_tool(state: State) -> str:

    print('======================= should_use_tool edge work')

    last: AIMessage = state["messages"][-1]

    # LangChain 原生解析
    if hasattr(last, "tool_calls") and last.tool_calls:
        print('=============== return use_tool 1')
        return "use_tool"

    # Fallback：原始文字偵測
    if isinstance(last.content, str) and parse_raw_tool_calls(last.content):
        print('=============== return use_tool 2')
        return "use_tool"

    print('=============== return END')
    return END


# ── 6. LangGraph ───────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges(
        "chatbot", should_use_tool, {"use_tool": "tool_node", END: END}
    )
    graph.add_edge("tool_node", "chatbot")
    return graph.compile()


# ── 7. 圖片處理 ────────────────────────────────────────────────────────────────
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif",  ".webp": "image/webp",  ".bmp": "image/bmp",
}


def encode_image_to_base64(image_path: str) -> tuple[str, str]:
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"不支援的圖片格式：{ext}")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, IMAGE_MEDIA_TYPES[ext]


def is_image_url(text: str) -> bool:
    if not re.match(r"^https?://", text, re.IGNORECASE):
        return False
    return os.path.splitext(text.split("?")[0])[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


def build_human_message(text: str, image_sources: list[str]) -> HumanMessage:
    if not image_sources:
        return HumanMessage(content=text)
    content: list[dict] = []
    for source in image_sources:
        if is_image_url(source):
            content.append({"type": "image_url", "image_url": {"url": source}})
        else:
            try:
                b64, mime = encode_image_to_base64(source)
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            except (ValueError, FileNotFoundError) as e:
                print(f"  ⚠️  圖片載入失敗（{source}）：{e}")
    if text:
        content.append({"type": "text", "text": text})
    return HumanMessage(content=content)


def parse_image_input() -> list[str]:
    print("  📎 圖片來源（本地路徑或 URL，多張請逐行輸入，留空結束）：")
    sources: list[str] = []
    while True:
        line = input("     > ").strip()
        if not line:
            break
        sources.append(line)
    return sources


# ── 8. 主程式 ──────────────────────────────────────────────────────────────────
def main():
    app = build_graph()
    conversation_history: list[BaseMessage] = [
        SystemMessage(content="你是一個專業且簡潔的AI助手，請用繁體中文回答, 簡短回答不要過度思考。")
    ]

    print("=" * 55)
    print("  Qwen3 聊天機器人（工具呼叫雙重偵測）")
    print("  輸入 'quit' / 'exit' 離開 | 'image' 附加圖片")
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

        image_sources: list[str] = []
        if user_input.lower() in {"image", "img", "圖片"}:
            image_sources = parse_image_input()
            try:
                user_input = input("  💬 訊息（可留空）：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n再見！")
                break

        conversation_history.append(build_human_message(user_input, image_sources))

        print('===================== main')

        result = app.invoke({"messages": conversation_history})

        ai_message: AIMessage = result["messages"][-1]

        print('=====================', ai_message)
        print('=====================')
        clean = clean_tool_call_tags(ai_message.content) if isinstance(ai_message.content, str) else ai_message.content
        print(f"\nQwen3：{clean}")

        conversation_history = result["messages"]

    try:
        app.get_graph().draw_png("workflow.png")
        print("\n📊 工作流程圖已儲存至 workflow.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()