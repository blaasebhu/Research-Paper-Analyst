import os, json, uuid
from typing import TypedDict, List, Dict, Any
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, END

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
PAPERS: Dict[str, Dict[str, Any]] = {}

def get_api_key(req=None):
    key = None
    if req and req.is_json:
        key = (req.get_json(silent=True) or {}).get("api_key")
    elif req:
        key = req.form.get("api_key") or req.headers.get("X-API-Key")
    return (key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()

def get_llm(api_key: str, temperature: float = 0.2):
    if not api_key: return None
    models = [os.environ.get("GEMINI_MODEL", ""), "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    for m in models:
        if not m: continue
        try: return ChatGoogleGenerativeAI(model=m, api_key=api_key, temperature=temperature)
        except Exception: continue
    return None

def get_embeddings(api_key: str):
    if not api_key: return None
    model_name = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    try: return GoogleGenerativeAIEmbeddings(model=model_name, api_key=api_key)
    except Exception: return None

@tool(description="Extracts raw text content from a PDF file path")
def paper_text_extractor(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(p.extract_text() or "" for p in reader.pages).strip()

@tool(description="Retrieves relevant text chunks from in-memory vector store matching query")
def document_retriever(query: str, chunks: List[str], api_key: str) -> str:
    if api_key:
        try:
            emb = get_embeddings(api_key)
            if emb:
                store = InMemoryVectorStore.from_texts(chunks, emb)
                docs = store.similarity_search(query, k=min(4, len(chunks)))
                return "\n\n".join(d.page_content for d in docs)
        except Exception: pass
    q_words = set(query.lower().split())
    ranked = sorted(chunks, key=lambda c: sum(1 for w in q_words if w in c.lower()), reverse=True)
    return "\n\n".join(ranked[:min(4, len(chunks))])

def extract_snippet(text: str, keywords: List[str], max_len: int = 240) -> str:
    tl = text.lower()
    for kw in keywords:
        pos = tl.find(kw)
        if pos != -1:
            snippet = text[pos:pos+max_len].replace("\n", " ").strip()
            if len(snippet) > 20: return snippet + "..."
    return "Not explicitly detailed in paper text"

def heuristic_summary(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 4]
    title = (lines[0][:117] + "...") if lines and len(lines[0]) > 120 else (lines[0] if lines else "Analyzed Paper")
    return {"title": title, "research_problem": extract_snippet(text, ["problem", "challenge", "motivation", "address"]), "objective": extract_snippet(text, ["objective", "aim", "goal", "in this paper we", "we propose"]), "methodology": extract_snippet(text, ["methodology", "method", "architecture", "framework", "approach"]), "dataset": extract_snippet(text, ["dataset", "benchmark", "corpus", "evaluated on", "data"]), "results": extract_snippet(text, ["result", "accuracy", "performance", "outperform", "achieves", "table"]), "limitations": extract_snippet(text, ["limitation", "drawback", "future work", "threat to validity"]), "conclusion": extract_snippet(text, ["conclusion", "conclude", "in summary", "summary"])}

def heuristic_questions(text: str) -> Dict[str, List[str]]:
    lines = [l.strip() for l in text.split("\n") if l.strip() and len(l.strip()) > 4]
    topic = (lines[0][:57] + "...") if lines and len(lines[0]) > 60 else (lines[0] if lines else "the proposed framework")
    return {
        "viva_questions": [f"What primary hypothesis or problem formulation motivates {topic}?", "How does the methodology mitigate baseline errors or computational complexity?", "What criteria governed the experimental setup, and how was bias prevented?", "How were empirical performance metrics and thresholds validated?", "What structural constraints or architectural trade-offs exist in this implementation?"],
        "presentation_questions": [f"Why is {topic} more practical than competing baseline approaches?", "What is the most significant measurable takeaway from the findings?", "How do environmental or domain shifts impact the stated results?", "What computational or memory resources are required during execution?", "How readily can this contribution generalize beyond the tested benchmarks?"],
        "future_research": ["Cross-benchmark validation on diverse domain distributions", "Algorithmic optimization to reduce latency and parameter overhead", "Ablation studies on architectural sub-components and hyperparameters", "Investigating robustness under noisy or out-of-distribution inputs"]
    }

@tool(description="Generates structured summary of research paper including title problem objective methodology dataset results limitations conclusion")
def paper_summary_generator(text: str, api_key: str) -> Dict[str, Any]:
    llm = get_llm(api_key, temperature=0.1)
    if llm:
        try:
            prompt = PromptTemplate.from_template(
                "Analyze this research paper text strictly based on the provided content. "
                "Never fabricate or invent authors, datasets, results, statistics, citations, or conclusions. "
                "If information is missing, state 'Not mentioned in paper'.\n"
                "Return ONLY a valid JSON object without markdown fences with keys:\n"
                "title, research_problem, objective, methodology, dataset, results, limitations, conclusion.\n\n"
                "Paper text:\n{text}"
            )
            chain = prompt | llm
            resp = chain.invoke({"text": text[:25000]})
            raw = resp.content.strip()
            if raw.startswith("```json"): raw = raw[7:-3].strip()
            elif raw.startswith("```"): raw = raw[3:-3].strip()
            data = json.loads(raw)
            if isinstance(data, dict) and "title" in data: return data
        except Exception: pass
    return heuristic_summary(text)

@tool(description="Generates viva questions presentation defense questions and future research directions")
def question_generator(text: str, api_key: str) -> Dict[str, List[str]]:
    llm = get_llm(api_key, temperature=0.2)
    if llm:
        try:
            prompt = PromptTemplate.from_template(
                "Based strictly on this paper text, provide academic interview and presentation preparation items. "
                "Never invent details not present in the text. Return ONLY a valid JSON object without markdown fences with keys:\n"
                "viva_questions (list of 5 rigorous viva questions with brief evaluation points),\n"
                "presentation_questions (list of 5 questions audience or reviewers will ask),\n"
                "future_research (list of 4 research directions based on paper limitations).\n\n"
                "Paper text:\n{text}"
            )
            chain = prompt | llm
            resp = chain.invoke({"text": text[:25000]})
            raw = resp.content.strip()
            if raw.startswith("```json"): raw = raw[7:-3].strip()
            elif raw.startswith("```"): raw = raw[3:-3].strip()
            data = json.loads(raw)
            if isinstance(data, dict) and "viva_questions" in data: return data
        except Exception: pass
    return heuristic_questions(text)

class AgentState(TypedDict, total=False):
    paper_path: str
    document_text: str
    chunks: List[str]
    question: str
    retrieved_context: str
    answer: str
    api_key: str
    mode: str
    summary: Dict[str, Any]
    viva_questions: List[str]
    presentation_questions: List[str]
    future_research: List[str]

def receive_paper(state: AgentState) -> Dict[str, Any]:
    return {"paper_path": state.get("paper_path", "")}

def extract_text(state: AgentState) -> Dict[str, Any]:
    text = state.get("document_text") or ""
    if not text and state.get("paper_path"):
        text = paper_text_extractor.invoke({"file_path": state["paper_path"]})
    return {"document_text": text}

def create_chunks(state: AgentState) -> Dict[str, Any]:
    chunks = state.get("chunks") or []
    if not chunks and state.get("document_text"):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunks = splitter.split_text(state["document_text"])
    return {"chunks": chunks}

def retrieve_relevant_content(state: AgentState) -> Dict[str, Any]:
    q = state.get("question", "")
    chunks = state.get("chunks", [])
    key = state.get("api_key", "")
    return {"retrieved_context": document_retriever.invoke({"query": q, "chunks": chunks, "api_key": key})}

def generate_answer(state: AgentState) -> Dict[str, Any]:
    llm = get_llm(state.get("api_key", ""), temperature=0.1)
    ctx = (state.get("retrieved_context") or "").strip()
    q = state.get("question", "")
    if not ctx: return {"answer": "The provided paper does not contain information to answer this question."}
    if llm:
        try:
            prompt = PromptTemplate.from_template("Academic assistant. Answer based ONLY on context below. If not found, say 'The provided paper does not contain information to answer this question.'\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
            resp = (prompt | llm).invoke({"context": ctx, "question": q})
            return {"answer": resp.content.strip()}
        except Exception as e:
            return {"answer": f"(Model notice: {e})\n\nGrounded excerpt from paper:\n{ctx[:400]}..."}
    return {"answer": f"Grounded excerpt from paper:\n\n{ctx[:600]}..."}

def generate_summary_node(state: AgentState) -> Dict[str, Any]:
    text = state.get("document_text", "")
    key = state.get("api_key", "")
    s = paper_summary_generator.invoke({"text": text, "api_key": key})
    q = question_generator.invoke({"text": text, "api_key": key})
    return {"summary": s, "viva_questions": q.get("viva_questions", []), "presentation_questions": q.get("presentation_questions", []), "future_research": q.get("future_research", [])}

def route_mode(state: AgentState) -> str:
    return "generate_summary_node" if state.get("mode") == "summary" else "retrieve_relevant_content"

builder = StateGraph(AgentState)
for n_name, n_fn in [("receive_paper", receive_paper), ("extract_text", extract_text), ("create_chunks", create_chunks), ("retrieve_relevant_content", retrieve_relevant_content), ("generate_answer", generate_answer), ("generate_summary_node", generate_summary_node)]:
    builder.add_node(n_name, n_fn)
for src, dst in [(START, "receive_paper"), ("receive_paper", "extract_text"), ("extract_text", "create_chunks"), ("retrieve_relevant_content", "generate_answer"), ("generate_answer", END), ("generate_summary_node", END)]:
    builder.add_edge(src, dst)
builder.add_conditional_edges("create_chunks", route_mode, {"generate_summary_node": "generate_summary_node", "retrieve_relevant_content": "retrieve_relevant_content"})
research_graph = builder.compile()

HTML_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Paper Assistant</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root { --bg: #0b0f19; --card: #131b2e; --card-border: #1e293b; --primary: #38bdf8; --primary-hover: #0284c7; --text: #f8fafc; --text-muted: #94a3b8; --accent: #818cf8; --accent-bg: rgba(129, 140, 248, 0.1); }
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
body { background: var(--bg); color: var(--text); line-height: 1.6; padding: 24px 16px; }
.container { max-width: 1200px; margin: 0 auto; }
header { text-align: center; margin-bottom: 24px; }
header h1 { font-size: 2rem; font-weight: 700; color: var(--primary); display: flex; align-items: center; justify-content: center; gap: 10px; }
header p { color: var(--text-muted); font-size: 0.95rem; margin-top: 4px; }
.api-bar { background: var(--card); border: 1px solid var(--card-border); padding: 12px 18px; border-radius: 12px; display: flex; gap: 12px; align-items: center; margin-bottom: 24px; }
.api-bar input { flex: 1; background: #0f172a; border: 1px solid var(--card-border); color: #fff; padding: 8px 12px; border-radius: 8px; font-size: 0.9rem; }
.grid { display: grid; grid-template-columns: 1fr 1.2fr; gap: 24px; }
@media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
.card { background: var(--card); border: 1px solid var(--card-border); border-radius: 16px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); margin-bottom: 24px; }
.card-title { font-size: 1.15rem; font-weight: 600; color: var(--primary); margin-bottom: 14px; display: flex; align-items: center; gap: 8px; }
.dropzone { border: 2px dashed var(--card-border); border-radius: 12px; padding: 24px; text-align: center; cursor: pointer; background: rgba(15, 23, 42, 0.5); }
.dropzone:hover { border-color: var(--primary); }
.dropzone input { display: none; }
.file-name { margin-top: 8px; font-size: 0.85rem; color: var(--accent); font-weight: 500; }
button { background: var(--primary); color: #0b0f19; font-weight: 600; border: none; border-radius: 8px; padding: 10px 18px; cursor: pointer; transition: background 0.2s; font-size: 0.9rem; }
button:hover { background: var(--primary-hover); }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-block { width: 100%; margin-top: 12px; }
.summary-item { margin-bottom: 10px; background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border-left: 3px solid var(--primary); }
.summary-item strong { display: block; font-size: 0.78rem; text-transform: uppercase; color: var(--primary); letter-spacing: 0.5px; margin-bottom: 4px; }
.summary-item p { font-size: 0.9rem; color: var(--text); }
.pill-list { list-style: none; display: flex; flex-direction: column; gap: 8px; }
.pill-list li { background: rgba(15, 23, 42, 0.6); border: 1px solid var(--card-border); padding: 10px 12px; border-radius: 8px; font-size: 0.88rem; }
.pill-list li span { color: var(--accent); font-weight: 600; margin-right: 6px; }
.qa-box { display: flex; flex-direction: column; gap: 12px; }
.qa-input-group { display: flex; gap: 8px; }
.qa-input-group input { flex: 1; background: #0f172a; border: 1px solid var(--card-border); color: #fff; padding: 10px 14px; border-radius: 8px; font-size: 0.9rem; }
.answer-box { background: rgba(15, 23, 42, 0.7); border: 1px solid var(--card-border); border-radius: 12px; padding: 16px; margin-top: 14px; }
.answer-text { font-size: 0.95rem; white-space: pre-wrap; margin-bottom: 10px; }
.context-details { font-size: 0.8rem; color: var(--text-muted); cursor: pointer; }
.context-content { margin-top: 8px; padding: 10px; background: #080c14; border-radius: 6px; font-family: monospace; white-space: pre-wrap; font-size: 0.75rem; display: none; }
.loader { display: none; margin: 12px 0; text-align: center; color: var(--primary); font-size: 0.9rem; font-weight: 500; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; background: var(--accent-bg); color: var(--accent); }
.sample-prompts { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.sample-btn { background: rgba(129, 140, 248, 0.15); color: var(--accent); border: 1px solid rgba(129, 140, 248, 0.3); font-size: 0.75rem; padding: 4px 10px; border-radius: 6px; }
.sample-btn:hover { background: rgba(129, 140, 248, 0.3); color: #fff; }
</style></head>
<body><div class="container"><header><h1><span>📄</span> Research Paper Assistant</h1><p>Lightweight LangGraph & RAG Analysis powered by Google Gemini</p></header>
<div class="api-bar"><span style="font-size:0.85rem; color:var(--text-muted); min-width:110px;">Gemini API Key:</span>
<input type="password" id="apiKey" placeholder="AIzaSy... (leave blank to use server environment key)"><span class="badge" id="keyBadge">{% if has_key %}ENV Key Active{% else %}Key Required{% endif %}</span></div>
<div class="grid"><div>
<div class="card"><div class="card-title"><span>📤</span> Upload Research Paper</div>
<div class="dropzone" id="dropzone" onclick="document.getElementById('pdfInput').click()"><input type="file" id="pdfInput" accept=".pdf"><p style="font-size:1.5rem; margin-bottom:6px;">📑</p><p><strong>Click to choose</strong> or drag & drop a PDF paper here</p><p class="file-name" id="fileName">No file selected</p></div>
<button class="btn-block" id="btnAnalyze" onclick="analyzePaper()">Analyze Paper</button>
<div class="loader" id="analyzeLoader">⏳ Extracting text, building chunks & analyzing...</div></div>
<div class="card" id="questionsCard" style="display:none;"><div class="card-title"><span>🎓</span> Viva & Presentation Questions</div>
<strong style="font-size:0.8rem; color:var(--accent); text-transform:uppercase;">Viva Defense Questions</strong><ul class="pill-list" id="vivaList" style="margin-top:6px; margin-bottom:16px;"></ul>
<strong style="font-size:0.8rem; color:var(--accent); text-transform:uppercase;">Presentation Questions</strong><ul class="pill-list" id="presList" style="margin-top:6px;"></ul></div>
<div class="card" id="futureCard" style="display:none;"><div class="card-title"><span>🚀</span> Future Research Directions</div><ul class="pill-list" id="futureList"></ul></div></div>
<div>
<div class="card" id="summaryCard" style="display:none;"><div class="card-title"><span>📊</span> Paper Analysis & Summary</div>
<div class="summary-item"><strong>Title</strong><p id="sTitle">-</p></div><div class="summary-item"><strong>Research Problem</strong><p id="sProblem">-</p></div>
<div class="summary-item"><strong>Objective</strong><p id="sObjective">-</p></div><div class="summary-item"><strong>Methodology</strong><p id="sMethodology">-</p></div>
<div class="summary-item"><strong>Dataset</strong><p id="sDataset">-</p></div><div class="summary-item"><strong>Results</strong><p id="sResults">-</p></div>
<div class="summary-item"><strong>Limitations</strong><p id="sLimitations">-</p></div><div class="summary-item"><strong>Conclusion</strong><p id="sConclusion">-</p></div></div>
<div class="card"><div class="card-title"><span>💬</span> Ask Paper (RAG Query)</div>
<div class="qa-box"><div class="qa-input-group"><input type="text" id="questionInput" placeholder="Ask anything about methodology, datasets, findings..."><button id="btnAsk" onclick="askPaper()">Ask Paper</button></div>
<div class="sample-prompts"><button class="sample-btn" onclick="setQuery('What dataset was used in this research?')">Dataset?</button><button class="sample-btn" onclick="setQuery('What is the core methodology proposed?')">Methodology?</button><button class="sample-btn" onclick="setQuery('What are the stated limitations of this work?')">Limitations?</button><button class="sample-btn" onclick="setQuery('What are the key numerical findings and results?')">Results?</button></div>
<div class="loader" id="askLoader">🔎 Retrieving relevant chunks & generating grounded answer...</div>
<div class="answer-box" id="answerBox" style="display:none;"><div style="font-size:0.8rem; color:var(--primary); font-weight:600; margin-bottom:6px;">GROUNDED ANSWER:</div><div class="answer-text" id="answerText"></div>
<div class="context-details" onclick="toggleContext()">▶ View Retrieved Chunks (In-Memory RAG Context)</div><div class="context-content" id="contextContent"></div></div></div></div></div></div></div>
<script>
let currentPaperId = null;
const fileInput = document.getElementById('pdfInput');
fileInput.addEventListener('change', (e) => { if (e.target.files.length) { document.getElementById('fileName').innerText = e.target.files[0].name; } });
function setQuery(q) { document.getElementById('questionInput').value = q; }
function toggleContext() { const el = document.getElementById('contextContent'); el.style.display = el.style.display === 'block' ? 'none' : 'block'; }
async function analyzePaper() {
    if (!fileInput.files.length) { alert('Please select a PDF file first.'); return; }
    const apiKey = document.getElementById('apiKey').value.trim();
    const formData = new FormData();
    formData.append('pdf', fileInput.files[0]);
    if (apiKey) formData.append('api_key', apiKey);
    document.getElementById('analyzeLoader').style.display = 'block';
    document.getElementById('btnAnalyze').disabled = true;
    try {
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) { alert('Error: ' + data.error); return; }
        currentPaperId = data.paper_id;
        const s = data.summary || {};
        [['Title','title'],['Problem','research_problem'],['Objective','objective'],['Methodology','methodology'],['Dataset','dataset'],['Results','results'],['Limitations','limitations'],['Conclusion','conclusion']].forEach(([id, k]) => { document.getElementById('s' + id).innerText = s[k] || '-'; });
        document.getElementById('summaryCard').style.display = 'block';
        const vList = document.getElementById('vivaList'); vList.innerHTML = '';
        (data.viva_questions || []).forEach((q, i) => { vList.innerHTML += `<li><span>Q${i+1}.</span> ${q}</li>`; });
        const pList = document.getElementById('presList'); pList.innerHTML = '';
        (data.presentation_questions || []).forEach((q, i) => { pList.innerHTML += `<li><span>Q${i+1}.</span> ${q}</li>`; });
        document.getElementById('questionsCard').style.display = 'block';
        const fList = document.getElementById('futureList'); fList.innerHTML = '';
        (data.future_research || []).forEach((r, i) => { fList.innerHTML += `<li><span>→</span> ${r}</li>`; });
        document.getElementById('futureCard').style.display = 'block';
    } catch (err) { alert('Failed to analyze paper: ' + err.message); }
    finally {
        document.getElementById('analyzeLoader').style.display = 'none';
        document.getElementById('btnAnalyze').disabled = false;
    }
}
async function askPaper() {
    if (!currentPaperId) { alert('Please upload and analyze a paper first.'); return; }
    const question = document.getElementById('questionInput').value.trim();
    if (!question) { alert('Please enter a question.'); return; }
    const apiKey = document.getElementById('apiKey').value.trim();
    document.getElementById('askLoader').style.display = 'block';
    document.getElementById('btnAsk').disabled = true;
    try {
        const res = await fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ paper_id: currentPaperId, question: question, api_key: apiKey }) });
        const data = await res.json();
        if (data.error) { alert('Error: ' + data.error); return; }
        document.getElementById('answerText').innerText = data.answer || 'No response generated.';
        document.getElementById('contextContent').innerText = data.retrieved_context || 'No context retrieved.';
        document.getElementById('answerBox').style.display = 'block';
    } catch (err) { alert('Failed to query paper: ' + err.message); }
    finally {
        document.getElementById('askLoader').style.display = 'none';
        document.getElementById('btnAsk').disabled = false;
    }
}
</script></body></html>"""

@app.route("/")
def index():
    has_key = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return render_template_string(HTML_TEMPLATE, has_key=has_key)

@app.route("/analyze", methods=["POST"])
def analyze():
    api_key = get_api_key(request)
    if not api_key:
        return jsonify({"error": "Gemini API key is required. Set GEMINI_API_KEY or provide it in the UI."}), 400
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided."}), 400
    file = request.files["pdf"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF document."}), 400
    paper_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_FOLDER, f"{paper_id}.pdf")
    file.save(save_path)
    try:
        reader = PdfReader(save_path)
        extracted = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        if not extracted:
            return jsonify({"error": "No selectable text found in the PDF. Scanned or image-only PDFs require OCR."}), 400
        result = research_graph.invoke({"paper_path": save_path, "document_text": extracted, "mode": "summary", "api_key": api_key})
        PAPERS[paper_id] = {
            "document_text": result.get("document_text", ""),
            "chunks": result.get("chunks", []),
            "summary": result.get("summary", {}),
            "viva_questions": result.get("viva_questions", []),
            "presentation_questions": result.get("presentation_questions", []),
            "future_research": result.get("future_research", [])
        }
        return jsonify({
            "paper_id": paper_id, "filename": file.filename,
            "page_count": len(reader.pages), "word_count": len(extracted.split()),
            "summary": result.get("summary", {}),
            "viva_questions": result.get("viva_questions", []),
            "presentation_questions": result.get("presentation_questions", []),
            "future_research": result.get("future_research", [])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    api_key = get_api_key(request)
    if not api_key:
        return jsonify({"error": "Gemini API key is required."}), 400
    data = request.get_json(silent=True) or {}
    paper_id = data.get("paper_id")
    question = (data.get("question") or "").strip()
    if not paper_id or paper_id not in PAPERS:
        return jsonify({"error": "Paper not found or session expired. Please upload and analyze a paper first."}), 400
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400
    paper = PAPERS[paper_id]
    try:
        result = research_graph.invoke({
            "document_text": paper["document_text"], "chunks": paper["chunks"],
            "question": question, "mode": "qa", "api_key": api_key
        })
        return jsonify({"answer": result.get("answer", ""), "retrieved_context": result.get("retrieved_context", "")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
