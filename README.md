# Research Paper Assistant 📄🔬

A lightweight, powerful academic research paper assistant built with **Python**, **Flask**, **Google Gemini**, **LangChain**, and **LangGraph**. It extracts text from uploaded PDF research papers, compiles comprehensive structured summaries, prepares academic defense/viva questions, suggests future research directions, and answers user questions grounded in the paper using lightweight in-memory RAG.

---

## 🌟 Key Features

1. **PDF Upload & Text Extraction**: Extracts clean text from research papers using `pypdf`.
2. **Structured Paper Summary**:
   - **Title**
   - **Research Problem**
   - **Objective**
   - **Key Methodology**
   - **Dataset Details**
   - **Empirical Results**
   - **Limitations**
   - **Conclusion**
3. **Academic Defense Preparation**:
   - **Viva Examination Questions**: 5 rigorous technical questions with evaluation points.
   - **Presentation Defense Questions**: 5 anticipated questions from conference or panel audiences.
4. **Future Research Directions**: Concrete next-step research ideas formulated from paper limitations.
5. **Grounded In-Memory RAG Q&A**:
   - Chunks paper content using `RecursiveCharacterTextSplitter`.
   - Embeds and indexes chunks in-memory using `InMemoryVectorStore` and Gemini Embeddings.
   - Retrieves top-k semantically relevant chunks for each user query.
   - Synthesizes grounded answers strictly from the retrieved context.
6. **Strict Grounding Rules**: Never fabricates authors, datasets, numbers, or citations. If information is not in the paper, it explicitly reports that it is not available.

---

## 🏗️ Architecture & LangGraph Flow

The assistant uses a compiled `langgraph.graph.StateGraph` implementing the following workflow:

```
                      ┌─────────┐
                      │  START  │
                      └────┬────┘
                           ▼
                  ┌─────────────────┐
                  │  Receive Paper  │
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │  Extract Text   │ (Tool: paper_text_extractor)
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │  Create Chunks  │ (RecursiveCharacterTextSplitter)
                  └────────┬────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      [mode == "summary"]          [mode == "qa"]
             │                           │
  ┌───────────────────────┐   ┌───────────────────────────┐
  │ Generate Summary Node │   │ Retrieve Relevant Content │ (Tool: document_retriever)
  │ (Tools: summary &     │   └─────────────┬─────────────┘
  │  question generators) │                 ▼
  └──────────┬────────────┘   ┌───────────────────────────┐
             │                │      Generate Answer      │
             │                └─────────────┬─────────────┘
             ▼                              ▼
          ┌─────┐                        ┌─────┐
          │ END │                        │ END │
          └─────┘                        └─────┘
```

### Registered Tools
- `paper_text_extractor`: Extracts text from the PDF file.
- `document_retriever`: Performs similarity retrieval over in-memory vector embeddings.
- `paper_summary_generator`: Generates 8-factor structured paper summary.
- `question_generator`: Produces viva questions, presentation defense questions, and future research paths.

---

## 🚀 Quick Start

### 1. Clone or Open the Project
```bash
cd "research paper analyst"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key
You can set your Gemini API key in an environment variable or `.env` file:
```bash
# In Windows PowerShell
$env:GEMINI_API_KEY="your-gemini-api-key-here"

# Or in Command Prompt
set GEMINI_API_KEY=your-gemini-api-key-here

# Or in Linux / macOS
export GEMINI_API_KEY="your-gemini-api-key-here"
```
*(Alternatively, you can enter the API key directly in the web UI header.)*

### 4. Run the Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 💻 How to Use

1. **Upload Paper**: Drag and drop or choose any research paper in `.pdf` format.
2. **Analyze Paper**: Click **Analyze Paper** to extract text, construct in-memory chunks, generate the 8-factor summary, viva questions, presentation questions, and future research ideas.
3. **Ask Paper**: Type any question into the input box or click one of the quick sample queries (e.g., *Dataset?*, *Methodology?*, *Limitations?*, *Results?*).
4. **Inspect Retrieved Context**: Click **View Retrieved Chunks** beneath any answer to inspect the exact passages retrieved from the in-memory vector store.

---

## 📁 Project Structure

Strictly adheres to the single-file constraint:

```
research paper analyst/
├── app.py              # Single self-contained Flask app with LangGraph, tools, RAG & UI (<= 400 lines)
├── requirements.txt    # Minimal required dependencies
└── README.md           # Documentation and usage guide
```
