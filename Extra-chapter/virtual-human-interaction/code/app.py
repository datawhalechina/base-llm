"""A small persona chat server with a scripted offline mode and an LLM adapter."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


PERSONA_PROMPT = """你叫小禾，是一位友善、耐心的中文 AI 學習夥伴。
你會用簡潔、自然的中文回應，先回答使用者的問題，再給一個可執行的下一步。
你不是人類，不聲稱擁有真實經歷；不確定時明確說明，不編造事實。
對話歷史只用於當前短期上下文；不要推斷使用者沒有明確表達的個人資訊。
"""


def _positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


MAX_HISTORY_TURNS = _positive_int_env("MAX_HISTORY_TURNS", 8)
MAX_INPUT_CHARS = _positive_int_env("MAX_INPUT_CHARS", 2000)
MODE = os.getenv("VIRTUAL_HUMAN_MODE", "demo").strip().lower()
if MODE not in {"demo", "api"}:
    MODE = "demo"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)
    message: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=80)


def _extract_name(messages: list[dict[str, str]]) -> str | None:
    patterns = (
        r"(?:我叫|我的名字是|請叫我|请叫我)\s*([\w\u4e00-\u9fff-]{1,20})",
        r"(?:我名字叫)\s*([\w\u4e00-\u9fff-]{1,20})",
    )
    for item in messages:
        if item["role"] != "user":
            continue
        for pattern in patterns:
            match = re.search(pattern, item["content"])
            if match:
                return match.group(1)
    return None


def _demo_reply(messages: list[dict[str, str]]) -> str:
    """Deterministic fallback used to demonstrate session behavior without a model."""
    current = messages[-1]["content"]
    earlier_user_messages = [
        item["content"] for item in messages[:-1] if item["role"] == "user"
    ]
    name = _extract_name(messages[:-1])

    if any(token in current for token in ("你叫什么", "你的名字", "你是谁")):
        return "我是小禾，你的 AI 學習夥伴。我會用簡潔的中文陪你拆解學習問題。"
    if any(token in current for token in ("我叫什么", "我的名字", "我名字")):
        if name:
            return f"記得，你之前說可以叫你{name}。我是小禾；我們可以接著聊你正在學的內容。"
        return "這段對話裡還沒有你的稱呼。你可以告訴我希望怎麼稱呼你。"
    if any(token in current for token in ("记得", "記得", "学什么", "學什麼")):
        if earlier_user_messages:
            previous = earlier_user_messages[-1].strip().replace("\n", " ")[:80]
            greeting = f"{name}，" if name else ""
            return f"{greeting}記得，你剛才提到：「{previous}」。我是小禾，可以沿著這個主題繼續聊。"
        return "目前還沒有較早的訊息可供回想；你可以先告訴我想學什麼。"
    if "刚才" in current or "剛才" in current or "上一条" in current:
        if earlier_user_messages:
            return f"你上一個提到的內容是：「{earlier_user_messages[-1]}」。我是小禾，可以接著幫你整理。"
        return "這是目前會話的第一個問題；你可以先告訴我想討論什麼。"
    if any(token in current for token in ("你好", "嗨", "哈囉", "您好")):
        greeting_name = _extract_name(messages) or name
        greeting = f"{greeting_name}，你好！" if greeting_name else "你好！"
        return f"{greeting}我是小禾，你的 AI 學習伙伴。今天想一起釐清哪個概念？"
    preview = current.strip().replace("\n", " ")[:80]
    return f"收到，你提到「{preview}」。我是小禾，可以先把它拆成概念、例子和下一步。你想從哪一部分開始？"


def _call_chat_api(messages: list[dict[str, str]]) -> str:
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "qwen2.5:3b")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 400,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        f"{base_url}/chat/completions", data=body, headers=headers, method="POST"
    )

    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("LLM service request failed") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("LLM service returned an unexpected response") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM service returned an empty response")
    return content.strip()


class ConversationManager:
    """In-memory session store; successful turns are committed as pairs."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._lock = threading.RLock()

    def reply(self, session_id: str, message: str) -> str:
        with self._lock:
            history = list(self._sessions.get(session_id, []))
            pending = history + [{"role": "user", "content": message}]
            bounded_history = pending[-(MAX_HISTORY_TURNS * 2 + 1) :]
            prompt = [{"role": "system", "content": PERSONA_PROMPT}, *bounded_history]
            try:
                answer = _demo_reply(bounded_history) if MODE == "demo" else _call_chat_api(prompt)
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=502,
                    detail="模型服務呼叫失敗，請檢查端點、模型名稱及本機服務狀態。",
                ) from exc

            committed = pending + [{"role": "assistant", "content": answer}]
            self._sessions[session_id] = committed[-(MAX_HISTORY_TURNS * 2) :]
            return answer

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


app = FastAPI(title="小禾：虛擬學習夥伴 Demo", version="1.0.0")
conversations = ConversationManager()


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {"mode": MODE, "max_history_turns": MAX_HISTORY_TURNS}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="訊息不能只有空白。")
    return {"reply": conversations.reply(request.session_id, message)}


@app.post("/api/reset")
def reset(request: ResetRequest) -> dict[str, bool]:
    conversations.reset(request.session_id)
    return {"ok": True}


@app.get("/api/session")
def new_session() -> dict[str, str]:
    return {"session_id": str(uuid4())}
