# FIFA_BOT — World Cup 2022 RAG Chatbot (FastAPI + LangChain + Ollama + FAISS)

A **Retrieval-Augmented Generation (RAG)** chatbot focused on **FIFA World Cup 2022** match facts, stats, and short recaps.

The bot:
- Loads structured match stats from `data/Fifa_world_cup_matches.csv`
- Loads additional match recap text files from `data/*.txt`
- Creates embeddings (via **Ollama embeddings**) and stores them in a **FAISS** vector database
- Uses a **local LLM via Ollama** to answer questions **only using retrieved context**
- Exposes a REST API using **FastAPI** (`POST /chat`) so any frontend can connect to it

---

## Features

- **RAG pipeline** — LangChain + FAISS vector store
- **Local-first** — powered by Ollama, no cloud API required by default
- **Grounded answers** — system prompt instructs the agent to:
  - always retrieve before answering
  - never invent facts or guess
  - respond with `No data available in the current database.` when nothing relevant is found
- **Conversation threads** — `thread_id` keeps each user's session separate (in-memory checkpointing via LangGraph)
- **FastAPI backend** with permissive CORS (easy to connect any frontend)

---

## Repository Structure

```
FIFA_BOT/
├── backend.py                   # FastAPI app (GET /, POST /chat)
├── main_func.py                 # ask_fifa_bot() helper for direct Python use
├── chatbot_ipynb_to_pyhton.py   # Core RAG pipeline: loads data, builds FAISS, creates agent
├── chatbot.ipynb                # Original Jupyter notebook (development scratchpad)
├── data.ipynb                   # Data exploration notebook
├── pyproject.toml               # Project dependencies (uv / pip)
├── uv.lock                      # Locked dependency versions
├── .python-version              # Python version pin
├── .env                         # Environment variables (currently empty)
└── data/
    ├── Fifa_world_cup_matches.csv   # Structured match stats
    └── *.txt                        # Match recap text files
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python >= 3.13 |
| API framework | FastAPI + Uvicorn |
| Orchestration | LangChain / LangGraph |
| Vector store | FAISS (`faiss-cpu`) |
| Embeddings | Ollama — `mxbai-embed-large:latest` |
| Chat model | Ollama — `llama3.2:3b` |
| Data handling | Pandas |

---

## Data Sources

All data lives inside `data/`:

1. **CSV match stats** — `data/Fifa_world_cup_matches.csv`  
   Each row is converted into a descriptive text document containing:
   - Teams and score
   - Match date and category (group stage / knockout)
   - Possession percentages
   - On-target attempts
   - Assists

2. **Text match recaps** — `data/*.txt`  
   All `.txt` files are recursively loaded and split into overlapping chunks (`chunk_size=500`, `chunk_overlap=100`) for better retrieval.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/APrem-7/FIFA_BOT.git
cd FIFA_BOT
```

### 2. Install Ollama

Download and install Ollama from [https://ollama.com](https://ollama.com), then pull the two models the project uses:

```bash
ollama pull mxbai-embed-large:latest
ollama pull llama3.2:3b
```

Make sure the Ollama server is running (`ollama serve`) before starting the bot.

### 3. Install Python dependencies

The repo ships with a `uv.lock` file. The recommended way is to use **uv**:

```bash
pip install uv          # if you don't have uv yet
uv sync
```

Alternatively, with plain pip:

```bash
pip install -e .
```

### 4. Confirm data files exist

```
data/Fifa_world_cup_matches.csv   ← required
data/*.txt                        ← at least one recap file recommended
```

---

## Running the FastAPI Backend

```bash
uvicorn backend:app --reload
```

The server starts on `http://127.0.0.1:8000` by default.

### Health check

```bash
curl http://127.0.0.1:8000/
```

Expected response:

```json
{"status": "ok", "message": "FIFA bot backend is running 🚀"}
```

### Query the `/chat` endpoint

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about the Argentina vs France World Cup Final", "thread_id": "user_1"}'
```

Example response:

```json
{
  "reply": "Argentina won the 2022 FIFA World Cup Final against France on penalties after a 3–3 draw..."
}
```

**Request body fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | string | ✅ | The question to ask the bot |
| `thread_id` | string | ❌ | Conversation session ID (default: `"default"`) |

---

## Using the Bot Directly from Python

The `ask_fifa_bot` helper in `main_func.py` lets you call the bot without the API server:

```python
from main_func import ask_fifa_bot

answer = ask_fifa_bot(
    "Tell me about the Argentina vs France Match in the World Cup Finals",
    thread_id="my_session"
)
print(answer)
```

You can also run `main_func.py` directly — it contains a sample call at the bottom:

```bash
python main_func.py
```

---

## How It Works (High-Level)

```
CSV rows ──► text Documents ──┐
                               ├──► FAISS vector store
txt files ─► chunks ──────────┘         │
                                         │  similarity search (top-5)
User question ──────────────────────────►│
                                         ▼
                               LangGraph agent (llama3.2:3b)
                               + strict system prompt
                                         │
                                         ▼
                               Grounded answer (or "No data available")
```

1. Each CSV row is turned into a descriptive text `Document`
2. All `.txt` files are loaded and split into chunks
3. Everything is embedded with `OllamaEmbeddings` (`mxbai-embed-large:latest`)
4. Embeddings are stored in FAISS
5. A retriever fetches the top-5 most similar documents per query
6. A LangGraph agent with a strict system prompt generates answers **only** from retrieved content

---

## Configuration

### Swap models

Open `chatbot_ipynb_to_pyhton.py` and change these two lines:

```python
# Embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

# Chat model
llm = ChatOllama(model="llama3.2:3b", temperature=0.2)
```

Any model available in your local Ollama installation can be used.

### CORS

`backend.py` currently allows all origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ← replace with your frontend URL in production
    ...
)
```

For production, replace `"*"` with your actual frontend domain, for example `["https://your-app.com"]`.

### Environment variables

The `.env` file is currently empty. If you add API keys or settings in the future:

- Commit `.env.example` with placeholder values
- Add `.env` to `.gitignore` so secrets are never committed

---

## Optional Improvements

Ideas to take the project further:

- **Persist the FAISS index** to disk (`vector_store.save_local(...)`) so it doesn't rebuild on every start
- **Add a `.gitignore`** to keep `__pycache__/`, `.DS_Store`, `.env`, and similar files out of the repo
- **Add `.env.example`** to document available configuration options
- **Write tests** — even a minimal smoke test for `GET /` and `POST /chat`
- **Dockerize** — add a `Dockerfile` and `docker-compose.yml` for FastAPI + Ollama
- **Streaming responses** — use FastAPI `StreamingResponse` with the Ollama streaming API for a better UX

---

## License

No license file is currently included. If you want others to use or contribute to this project, consider adding an open-source license such as [MIT](https://choosealicense.com/licenses/mit/) or [Apache 2.0](https://choosealicense.com/licenses/apache-2.0/).

---

## Acknowledgements

Built with:
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://python.langchain.com/) / [LangGraph](https://langchain-ai.github.io/langgraph/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)

