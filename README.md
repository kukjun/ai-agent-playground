# AI Agent Playground

다양한 AI Agent 프레임워크를 테스트하기 위한 공간입니다.

## 프로젝트 구조

```
ai-agent-playground/
├── README.md
├── .env                 # 환경변수 (API keys)
├── .env.example         # 환경변수 예시
├── pyproject.toml       # 패키지 관리 (uv)
├── langchain/
│   └── examples/        # LangChain 예제 노트북
├── langgraph/
│   └── examples/        # LangGraph 예제 노트북
└── crew-ai/
    └── examples/        # CrewAI 예제 노트북
```

## 설치

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치
uv sync
```

## 환경변수 설정

`.env.example`을 복사하여 `.env` 파일을 생성하고 필요한 값을 입력하세요.

```bash
cp .env.example .env
```

## 지원 프레임워크

- **LangChain**: LLM 애플리케이션 개발 프레임워크
- **LangGraph**: 상태 기반 멀티 액터 애플리케이션 구축
- **CrewAI**: 멀티 에이전트 협업 프레임워크
