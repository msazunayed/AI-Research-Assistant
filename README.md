# Research Desk - AI Research Assistant

Give it a topic. It searches the web, reads the sources, and writes a cited report - streamed live to a web UI.

## Project layout

```
AI-Research-Assistant/
├── backend/
│   └── main.py            FastAPI app + pipeline
├── frontend/
│   └── index.html         Web UI (no build step)
├── .env                   Your API keys (git-ignored, create it yourself)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Setup

1. **LLM key** — free key from [console.groq.com](https://console.groq.com) (default), or any OpenAI-compatible provider (xAI, Ollama, etc.)
2. **Search key** — free key from [tavily.com](https://tavily.com)
3. Create `.env` at the repo root:
   ```bash
   "You can use any free like: GROK API KEY"
   OPENAI_API_KEY=your-groq-key
   TAVILY_API_KEY=your-tavily-key
   ```
   (Everything else has a default — see `main.py` if you want to tweak models, concurrency, retries, etc.)

## Run

```bash
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in your browser.

## Using it

Type a topic, click **Start research**, watch it plan queries → search → read sources → write the report, complete with `[1]`-style citations.

## Customizing the report

Report tone/structure comes from the `write_report` prompt in `backend/main.py` — edit it to change the style.

## Output 
https://github.com/user-attachments/assets/e44f65bc-ac34-4ff3-81ec-b4942b6aee2b



