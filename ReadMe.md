# **QuantaBot: Advanced LangChain RAG System**

QuantaBot is a sophisticated Python application designed to handle large documents (PDF/TXT/DOCX) for in-depth research discussions. It specializes in working with research papers or lengthy documents, providing precise and contextually relevant response to user questions based on document content(s), along with citations. Initially, it was built to aid researchers/professors associated in Department of Dermatology at the University of Michigan.

**FILE STRUCTURE:**

```
QuantaBot/
  ├── scripts/
  │   ├── run_app.py                         # Streamlit entry point
  │   └── run_medqa_eval.py                  # MedQA benchmark entry point
  ├── src/
  │   └── quantabot/                         # Importable package
  │       ├── core/
  │       │   ├── rag.py                     # LangChain RAG system
  │       │   └── llm.py                     # OpenAI wrapper
  │       ├── loaders/                       # PDF / TXT / DOCX loaders
  │       ├── utils/
  │       │   ├── document_processor.py
  │       │   ├── embedding_wrapper.py
  │       │   └── text_processing.py
  │       └── ui/
  │           └── streamlit_app.py
  ├── evaluations/
  │   ├── README.md
  │   └── medqa/
  │       └── evaluator.py                   # USMLE / MedQA evaluator
  ├── data/                                  # MedQA dataset + medical textbooks
  ├── results/                               # Eval JSON / CSV outputs
  ├── docs/                                  # Research roadmap, notes
  └── archive/                               # Prior experiments
```

### Core Technologies

- **Python** as the code language.
- **BERT-based embeddings** (via Sentence-BERT from HuggingFace).
- **GPT-4** as the response generator (via OpenAI API).
- **ChromaDB** for fast vector storage and retrieval.
- **Streamlit** for frontend deployment.
- **LangChain** for orchestrating query pipelines.

### Advanced LangChain Features

#### Retriever Types

- **Small documents** (< 20 chunks): Uses ContextualCompressionRetriever
- **Large documents** (>= 20 chunks): Uses EnsembleRetriever with BM25

- **MultiQueryRetriever**
  - Automatic query expansion using GPT-4
- **EnsembleRetriever**
  - Hybrid semantic (vector embeddings) + BM25 keyword search
- **ParentDocumentRetriever**
  - Retrieves small, focused chunks for relevance
  - Provides large parent documents for full context
- **CrossEncoderReranker**
  - Advanced relevance scoring using dedicated reranking model
  - More accurate than simple cosine similarity
  - Optimizes final document selection
- **ConversationMemory**
  - Professional conversation context management
  - Maintains relevant history while managing token limits
  - Enables natural follow-up questions
- **Advanced Monitoring**
  - Built-in callbacks for performance tracking
  - Detailed retrieval statistics
  - Real-time system monitoring
- **MPNet Embeddings**
  - Up to 768-dimensional embeddings

> This project separates retrieval and generation.

> Historical experiments and engineering notes from earlier iterations live in `archive/`.

### Configurable embedding model

The embedding model defaults to `all-mpnet-base-v2` but can be overridden via env var:

```bash
export QUANTA_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

## **Installation**

Follow the steps below to set up your chatbot:

1. **Install Python**

   If you have never used Python on your computer before, make sure to download one.

2. **Install dependencies**

   Run the following command to install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**

   Obtain a personal API key from [OpenAI](https://openai.com/index/openai-api/) and add it to your terminal environment, like this:

   (For temporary usage, please contact the email at the very bottom.)

- **macOS / Linux**:

  ```bash
  export OPENAI_API_KEY='YOUR_OPEN_AI_KEY'
  ```

- **Windows**:

  ```bash
  set OPENAI_API_KEY='YOUR_OPEN_AI_KEY'
  ```

## **Usage**

To use Quanta Chatbot, follow these steps:

1. Ensure all dependencies are installed and OpenAI API key is incorporated into the file.
2. Launch the Streamlit app from the QuantaBot project root:

   ```bash
   streamlit run scripts/run_app.py
   ```

   This launches the **Advanced LangChain RAG System** interface.

3. A page will be launched on your default web browser.

   <u>**NOTICE**</u>:

   - At this stage, if the page shows a red screen with a _"ModuleNotFound"_ error, please **install** the ones listed in the error message manually at the terminal as well, using the following template. This tends to vary by computer. Then, repeat step 2.
     ```bash
     pip install <dependencies you need>
     ```
   - If _"StreamlitDuplicateElementKey"_ error pops up, just **refresh** the page once more.

4. Upload as many files as you want by dragging or selecting.
   - It is intended that files MUST BE uploaded FIRST for program to run.
5. Ask questions in English about the loaded files using the chat interface.

## **Resolving Conflicts**

While running the program, various errors might occur. If so, make sure to try these temporary solutions.

1. Try pip -> pip3 / python -> python3
2. Update the problem dependencies.

   ```bash
   pip install --upgrade <dependency being updated>
   ```

3. Ensure that your OpenAI has enough credit balance, else it won't be making successful API calls. This can be checked [here.](platform.openai.com)
4. Make sure to search up the error message that appears on the terminal window.
5. Hand type the terminal commands, instead of copying and pasting.

## **[Testing] Model Evaluation**

Earlier component-level retrieval experiments (TF-IDF vs SBERT, BEIR/NFCorpus benchmarks) are archived in `archive/` for historical reference.

The active benchmark is **MedQA (USMLE)**:

```bash
python scripts/run_medqa_eval.py
```

The evaluator reports:
- **Letter accuracy** — strict A/B/C/D/E match (the headline metric)
- **Retrieval-contains-answer rate** — diagnostic for whether retrieval or the LLM is the bottleneck
- Loose substring matches, response time, and clinical-relevance heuristics

See `evaluations/README.md` and `docs/research_roadmap.md` for details.

##

For further inquiries, please feel free to contact dongeunk@umich.edu.
