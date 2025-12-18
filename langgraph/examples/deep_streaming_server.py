"""
Deep Socket.IO Streaming Server

LangGraph의 다중 노드 그래프를 실행하고,
각 단계별 이벤트를 Socket.IO로 스트리밍합니다.

실행: python deep_streaming_server.py
테스트: http://localhost:8000
"""
import os
import uuid
import asyncio
from datetime import datetime
from typing import TypedDict, List

import socketio
from aiohttp import web
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

load_dotenv()

# ============================================================
# 설정
# ============================================================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("STREAMING_MODEL", "gemma3:12b")

# Socket.IO 서버
sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

# LLM
llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_URL, temperature=0.7)


# ============================================================
# Mock Database
# ============================================================
class MockDatabase:
    def __init__(self):
        self.records: List[dict] = []

    async def save(self, data: dict) -> dict:
        await asyncio.sleep(0.3)  # DB 지연 시뮬레이션
        record = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            **data
        }
        self.records.append(record)
        return record

    def get_all(self) -> List[dict]:
        return self.records


mock_db = MockDatabase()


# ============================================================
# Graph State & Nodes
# ============================================================
class GraphState(TypedDict):
    user_input: str
    analysis: str
    response: str
    saved_record: dict


async def analyzer_node(state: GraphState) -> GraphState:
    """[Node 1] 사용자 입력 분석"""
    prompt = f"""다음 입력을 분석해주세요.
- 주요 의도
- 핵심 키워드
- 감정 톤

입력: {state["user_input"]}

간단히 분석해주세요."""

    response = await llm.ainvoke(prompt)
    return {"analysis": response.content}


async def generator_node(state: GraphState) -> GraphState:
    """[Node 2] 응답 생성"""
    prompt = f"""분석 결과를 바탕으로 응답을 생성해주세요.

원본 입력: {state["user_input"]}

분석 결과:
{state["analysis"]}

친절하고 도움이 되는 응답을 작성해주세요."""

    response = await llm.ainvoke(prompt)
    return {"response": response.content}


async def saver_node(state: GraphState) -> GraphState:
    """[Node 3] DB 저장"""
    record = await mock_db.save({
        "user_input": state["user_input"],
        "analysis": state["analysis"],
        "response": state["response"],
    })
    return {"saved_record": record}


# 그래프 빌드
workflow = StateGraph(GraphState)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("generator", generator_node)
workflow.add_node("saver", saver_node)
workflow.add_edge(START, "analyzer")
workflow.add_edge("analyzer", "generator")
workflow.add_edge("generator", "saver")
workflow.add_edge("saver", END)
graph = workflow.compile()

# 추적할 노드 이름
NODE_NAMES = {"analyzer", "generator", "saver"}


# ============================================================
# Socket.IO Events
# ============================================================
@sio.event
async def connect(sid, environ):
    print(f"[연결] {sid}")
    await sio.emit("connected", {"sid": sid}, to=sid)


@sio.event
async def disconnect(sid):
    print(f"[연결 해제] {sid}")


@sio.event
async def chat(sid, data):
    """채팅 메시지 처리 - 전체 이벤트 스트리밍"""
    user_input = data.get("message", "")
    print(f"[{sid}] 입력: {user_input}")

    current_node = None

    try:
        async for event in graph.astream_events(
            {"user_input": user_input},
            version="v2"
        ):
            kind = event["event"]

            # 노드 시작
            if kind == "on_chain_start":
                node_name = event.get("name", "")
                if node_name in NODE_NAMES:
                    current_node = node_name
                    print(f"  [NODE START] {node_name}")
                    await sio.emit("node_start", {"node": node_name}, to=sid)

            # 노드 종료
            elif kind == "on_chain_end":
                node_name = event.get("name", "")
                if node_name in NODE_NAMES:
                    output = event.get("data", {}).get("output", {})
                    print(f"  [NODE END] {node_name}")
                    await sio.emit("node_end", {
                        "node": node_name,
                        "output": str(output)[:500]
                    }, to=sid)

                    # DB 저장 이벤트
                    if node_name == "saver" and "saved_record" in output:
                        print(f"  [DB SAVE] {output['saved_record']['id']}")
                        await sio.emit("db_save", output["saved_record"], to=sid)

            # LLM 토큰 스트리밍
            elif kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and chunk.content:
                    await sio.emit("token", {
                        "node": current_node,
                        "content": chunk.content
                    }, to=sid)

        # 완료
        print(f"  [DONE]")
        await sio.emit("done", {"success": True}, to=sid)

    except Exception as e:
        print(f"  [ERROR] {e}")
        await sio.emit("error", {"message": str(e)}, to=sid)


# ============================================================
# HTML 테스트 페이지
# ============================================================
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Deep Streaming Test</title>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Consolas', 'Monaco', monospace;
            padding: 20px;
            background: #0d1117;
            color: #c9d1d9;
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 { color: #58a6ff; }
        #output {
            background: #161b22;
            padding: 20px;
            border-radius: 8px;
            height: 500px;
            overflow-y: auto;
            border: 1px solid #30363d;
            margin-bottom: 20px;
        }
        .node-start { color: #3fb950; font-weight: bold; margin-top: 15px; }
        .node-end { color: #58a6ff; }
        .token { color: #c9d1d9; }
        .db-save { color: #d29922; font-weight: bold; }
        .done { color: #a371f7; font-weight: bold; margin-top: 15px; }
        .error { color: #f85149; }
        .send { color: #8b949e; margin-top: 20px; border-top: 1px solid #30363d; padding-top: 10px; }

        .input-container { display: flex; gap: 10px; }
        input {
            flex: 1;
            padding: 12px;
            font-size: 16px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
        }
        input:focus { outline: none; border-color: #58a6ff; }
        button {
            padding: 12px 24px;
            font-size: 16px;
            cursor: pointer;
            background: #238636;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: bold;
        }
        button:hover { background: #2ea043; }

        .status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status.connected { background: #3fb950; }
        .status.disconnected { background: #f85149; }
    </style>
</head>
<body>
    <h1><span class="status" id="status"></span>Deep Socket.IO Streaming</h1>
    <p>LangGraph 노드별 이벤트와 LLM 토큰을 실시간으로 확인합니다.</p>

    <div id="output"></div>

    <div class="input-container">
        <input type="text" id="input" placeholder="메시지를 입력하세요..." />
        <button onclick="sendMessage()">전송</button>
    </div>

    <script>
        const socket = io();
        const output = document.getElementById('output');
        const status = document.getElementById('status');
        let currentTokenDiv = null;

        function log(msg, cls = '') {
            currentTokenDiv = null;  // 새 로그면 토큰 div 리셋
            const div = document.createElement('div');
            div.className = cls;
            div.innerHTML = msg;
            output.appendChild(div);
            output.scrollTop = output.scrollHeight;
        }

        function appendToken(content) {
            if (!currentTokenDiv) {
                currentTokenDiv = document.createElement('div');
                currentTokenDiv.className = 'token';
                output.appendChild(currentTokenDiv);
            }
            currentTokenDiv.innerHTML += content;
            output.scrollTop = output.scrollHeight;
        }

        socket.on('connect', () => {
            status.className = 'status connected';
        });

        socket.on('disconnect', () => {
            status.className = 'status disconnected';
        });

        socket.on('connected', (data) => {
            log(`✅ 연결됨: ${data.sid}`, 'node-end');
        });

        socket.on('node_start', (data) => {
            log(`🚀 [${data.node.toUpperCase()}] 시작`, 'node-start');
        });

        socket.on('node_end', (data) => {
            log(`✅ [${data.node.toUpperCase()}] 완료`, 'node-end');
        });

        socket.on('token', (data) => {
            appendToken(data.content);
        });

        socket.on('db_save', (data) => {
            log(`💾 [DB 저장] ID: ${data.id}`, 'db-save');
            log(`   생성일: ${data.created_at}`, 'db-save');
        });

        socket.on('done', () => {
            log(`🎉 전체 완료!`, 'done');
        });

        socket.on('error', (data) => {
            log(`❌ 오류: ${data.message}`, 'error');
        });

        function sendMessage() {
            const input = document.getElementById('input');
            const message = input.value.trim();
            if (message) {
                log(`📤 입력: ${message}`, 'send');
                socket.emit('chat', { message });
                input.value = '';
            }
        }

        document.getElementById('input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });

        // 초기 상태
        status.className = 'status disconnected';
    </script>
</body>
</html>
"""


async def index(request):
    return web.Response(text=HTML_PAGE, content_type="text/html")


app.router.add_get("/", index)


# ============================================================
# 서버 실행
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Deep Socket.IO Streaming Server")
    print("=" * 60)
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Model: {MODEL_NAME}")
    print()
    print("Graph: START → analyzer → generator → saver → END")
    print()
    print("🚀 서버 시작: http://localhost:8000")
    print("=" * 60)
    web.run_app(app, host="0.0.0.0", port=8000)
