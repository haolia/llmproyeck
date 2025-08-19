# chatbot_app.py
import os
import time
import math
import json
import datetime as dt
from io import BytesIO
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# --------- LangChain / RAG imports ----------
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

APP_TITLE = "Gemini RAG Chatbot — Final Project"
DEFAULT_MODEL = "gemini-1.5-flash"
DOC_DIR = "data"
VECTOR_STORE_PATH = os.path.join(DOC_DIR, "faiss_index")

# ---------------- Utilities -----------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_api_key_from_env() -> str:
    load_dotenv()
    return os.getenv("GOOGLE_API_KEY", "")

def configure_gemini(api_key: str, model_name: str, temperature: float):
    genai.configure(api_key=api_key)
    safety_settings = [
        # keep defaults; could be adjusted per ethics policy
    ]
    generation_config = {
        "temperature": float(temperature),
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    model = genai.GenerativeModel(model_name, generation_config=generation_config, safety_settings=safety_settings)
    return model

# ----------------- RAG ----------------------
def load_documents_from_folder(folder: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(folder):
        return docs
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if fname.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(path)
                docs.extend(loader.load())
            except Exception as e:
                st.warning(f"Gagal memuat PDF: {fname} — {e}")
        elif fname.lower().endswith((".txt", ".md")):
            try:
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                st.warning(f"Gagal memuat teks: {fname} — {e}")
    return docs

def build_vectorstore(docs: List[Document], api_key: str) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/text-embedding-004")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def save_vectorstore(vs: FAISS, path: str):
    vs.save_local(path)

def load_vectorstore(path: str, api_key: str) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/text-embedding-004")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def retriever_search(vs: FAISS, query: str, k: int = 4) -> List[Tuple[str, float]]:
    results = vs.similarity_search_with_score(query, k=k)
    return [(doc.page_content, score) for doc, score in results]

# ------------- Tools / Function Calling -------------
def tool_calculator(expression: str) -> str:
    """Safe calculator: supports + - * / and parentheses."""
    try:
        # Very basic safe eval
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            return "Error: hanya angka dan + - * / ( ) yang diperbolehkan."
        result = eval(expression, {"__builtins__": {}})
        return f"Hasil: {result}"
    except Exception as e:
        return f"Error kalkulasi: {e}"

def tool_now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def tool_docsearch(vs: FAISS, query: str) -> str:
    if vs is None:
        return "Index belum dibuat. Unggah dokumen terlebih dahulu."
    hits = retriever_search(vs, query, k=4)
    lines = []
    for i, (content, score) in enumerate(hits, 1):
        lines.append(f"[{i}] score={score:.3f}\n{content[:700]}")
    return "\n\n".join(lines) if lines else "Tidak ditemukan."

TOOLS_SPEC = [
    {
        "name": "calculator",
        "description": "Evaluate a basic math expression with + - * / and parentheses.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate."}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "now",
        "description": "Get current local datetime as a string.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "docsearch",
        "description": "Search the local RAG index for relevant passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query over the indexed documents."}
            },
            "required": ["query"]
        }
    }
]

def call_tool(tool_name: str, args: dict, vs: FAISS) -> str:
    if tool_name == "calculator":
        return tool_calculator(args.get("expression", ""))
    if tool_name == "now":
        return tool_now()
    if tool_name == "docsearch":
        return tool_docsearch(vs, args.get("query", ""))
    return f"Tool {tool_name} tidak dikenal."

# --------------- Streamlit UI -----------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts: {"role": "user"/"assistant", "content": str}
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "mode" not in st.session_state:
        st.session_state.mode = "General Chat"

def sidebar_controls():
    st.sidebar.title("⚙️ Controls")
    api_key = st.sidebar.text_input("GOOGLE_API_KEY", value=st.session_state.api_key or load_api_key_from_env(), type="password")
    st.session_state.api_key = api_key.strip()
    model_name = st.sidebar.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"], index=0)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
    mode = st.sidebar.radio("Mode", ["General Chat", "RAG over Docs", "Tools/Agents"])
    st.session_state.mode = mode

    if st.sidebar.button("🔁 Reset Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Upload Documents (PDF/TXT/MD)")
    uploads = st.sidebar.file_uploader("Drop files", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if uploads:
        ensure_dir(DOC_DIR)
        for f in uploads:
            path = os.path.join(DOC_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
        st.sidebar.success(f"Uploaded {len(uploads)} file(s). Click **Rebuild Index**.")

    if st.sidebar.button("🧱 Rebuild Index"):
        with st.spinner("Building FAISS index..."):
            docs = load_documents_from_folder(DOC_DIR)
            if not docs:
                st.sidebar.warning("Tidak ada dokumen di folder data/.")
            else:
                vs = build_vectorstore(docs, st.session_state.api_key)
                save_vectorstore(vs, VECTOR_STORE_PATH)
                st.session_state.vectorstore = vs
                st.sidebar.success("Index berhasil dibuat.")

    # Try auto-load existing index
    if st.session_state.vectorstore is None and os.path.isdir(VECTOR_STORE_PATH):
        try:
            st.session_state.vectorstore = load_vectorstore(VECTOR_STORE_PATH, st.session_state.api_key)
        except Exception:
            pass

    return model_name, temperature

def render_header():
    st.title(APP_TITLE)
    st.caption("Showcasing: Gemini prompting • RAG • tools/agents • Streamlit UI")

def render_chat_history():
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

def llm_generate(model, prompt: str, system: str = None) -> str:
    if system:
        parts = [{"text": system}, {"text": prompt}]
        resp = model.generate_content(parts)
    else:
        resp = model.generate_content(prompt)
    return resp.text if hasattr(resp, "text") else str(resp)

def llm_generate_with_tools(model, prompt: str, vs: FAISS) -> str:
    # Define tools and let the model decide
    response = model.generate_content(
        prompt,
        tools=TOOLS_SPEC
    )
    # If the model requests a tool call, handle it
    try:
        for part in (response.candidates[0].content.parts or []):
            if hasattr(part, "function_call") and part.function_call is not None:
                name = part.function_call.name
                args = {k: v for k, v in (part.function_call.args.items() if hasattr(part.function_call, "args") else {})}
                tool_result = call_tool(name, args, vs)
                # Send tool result back to the model for final answer
                followup = model.generate_content([
                    {"role": "user", "parts": [{"text": prompt}]},
                    {"role": "tool", "parts": [{"text": f"Tool {name} result:\n{tool_result}"}]}
                ])
                return followup.text or tool_result
    except Exception as e:
        return f"(Tool error) {e}\n\nModel reply: {getattr(response, 'text', '')}"
    return getattr(response, "text", "") or "(no response)"

def main():
    init_session_state()
    render_header()
    model_name, temperature = sidebar_controls()

    if not st.session_state.api_key:
        st.info("Masukkan **GOOGLE_API_KEY** di sidebar atau file `.env`.")
        return

    model = configure_gemini(st.session_state.api_key, model_name, temperature)

    render_chat_history()
    user_msg = st.chat_input("Tanya sesuatu…")
    if user_msg is None:
        return

    st.session_state.messages.append({"role": "user", "content": user_msg})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            mode = st.session_state.mode
            if mode == "General Chat":
                system = "You are a helpful, concise assistant that follows ethical AI guidelines."
                answer = llm_generate(model, user_msg, system=system)
            elif mode == "RAG over Docs":
                if st.session_state.vectorstore is None:
                    answer = "Index belum dibuat. Unggah file di sidebar lalu klik **Rebuild Index**."
                else:
                    context_chunks = retriever_search(st.session_state.vectorstore, user_msg, k=4)
                    context_text = "\n\n".join([c for c, _ in context_chunks])
                    prompt = f"Jawablah dengan merujuk konteks berikut jika relevan. Jika tidak ada di konteks, katakan tidak yakin.\n\nKONTEKS:\n{context_text}\n\nPERTANYAAN:\n{user_msg}"
                    answer = llm_generate(model, prompt)
            else:  # Tools/Agents
                answer = llm_generate_with_tools(model, user_msg, st.session_state.vectorstore)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    ensure_dir(DOC_DIR)
    main()
