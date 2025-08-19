# Final Project: Gemini RAG Chatbot (Streamlit + LangChain)

A single-file Streamlit app that showcases the full learning path:
- **Intro & Ethics**: prompt templates and safety notes.
- **Implementing Generative AI with Gemini**: advanced prompting, config, and function calling.
- **RAG with LangChain**: local PDF ingestion, FAISS vector DB, retrieval-augmented answers.
- **Building Apps with LLM**: Streamlit UI, simple tools/agents, and deployment tips.

## Features
- Chat with **Gemini** in 3 modes: *General Chat*, *RAG over Docs*, *Tools/Agents*.
- **Upload PDFs** → auto-index with **FAISS** (LangChain) and **Google Generative AI Embeddings**.
- **Function calling/tools**: calculator, current time, and doc-search as callable tools.
- Sidebar **API Key** input (or `.env`), model/temperature controls, and session reset.
- Clean, responsive UI with chat history.

## Quickstart (macOS & VS Code)
1. **Create/activate venv**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install deps**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set your API key**
   - Copy `.env.example` → `.env`, then add your Google key.
   - Or set directly in the app sidebar at runtime.
4. **Run**
   ```bash
   streamlit run chatbot_app.py
   ```
5. **Open the URL** shown in the terminal (usually `http://localhost:8501`).

### VS Code tips
- Use the bottom-left Python interpreter selector → pick `.venv`.
- Run from VS Code terminal (**Terminal → New Terminal**) after activating `.venv`.
- If you see “ModuleNotFoundError”, ensure the terminal uses the same interpreter as VS Code.

### If the URL opens to a blank page
- Wait ~5 seconds—first index build can take time.
- Check the terminal for errors.
- Ensure only **one** Streamlit run on port 8501. If busy, run with `streamlit run chatbot_app.py --server.port 8502`.
- Try disabling corporate/VPN proxy or adblockers for `localhost`.

## Deploy (Streamlit Community Cloud)
- Push this folder to GitHub.
- In Streamlit Cloud → New app → select repo and `chatbot_app.py`.
- Add **GOOGLE_API_KEY** as a secret in *App settings → Secrets*.
- Optional: pre-load PDFs by putting them in the `data/` folder in the repo.

## Folder
```
final_gemini_chatbot/
├─ chatbot_app.py
├─ requirements.txt
├─ .env.example
├─ README.md
└─ data/            # put your PDFs here
```
