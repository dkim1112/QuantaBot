# Problems I Faced & Approaches (Non-Exhaustive List):

## Problem 1: Persistent Confusion from Old Documents

### 🔍 The Problem

One crucial issue I consistently encountered was this:

> _"Even when I upload a new document, my chatbot still references information from documents I uploaded a long time ago. It gets confused and pulls irrelevant answers from older PDFs, completely ignoring the newly uploaded one."_

This behavior was especially problematic when testing or iterating over the system multiple times - responses were often inconsistent, outdated, or completely off-topic (not getting information from what I just uploaded.)

### Root Cause

After some digging, I discovered the core issue was **how ChromaDB handles persistence**.

- Every uploaded document was being added to the **same persistent ChromaDB directory**:
  ```python
  Chroma(persist_directory="chroma_db", ...)
  ```
- This meant that **all documents - old PDFs, new uploads, test files - coexisted in a single vector store**.
- As a result, the retriever had no idea which document was "current" or more relevant.
- When using **cosine similarity + top-k retrieval**, older documents sometimes ranked higher, even if they were from a totally unrelated topic.

This polluted the prompt sent to GPT, causing:

- Confusing or irrelevant answers
- Hallucinations
- Mismatches between the user’s query and the actual source context

### Why I Initially Designed It This Way?

My intention was to **retain memory across uploads**.  
I wanted the system to:

- Recall from past papers **if** needed
- Enhance response quality by providing a **larger knowledge base**

But in practice, this conflicted with the current retrieval (top-k) method:

- Cosine similarity isn't perfect - sometimes irrelevant chunks get high similarity scores, considering when they were calculated (under what context).
- Without document-level filtering, this increased noise and reduced accuracy.

### ✅ How I Solved It

To fix this issue, I made the system **reset the vector database** before each new upload (session).

- The solution: **Clear the ChromaDB vector store before every new session is processed**

As a result,

- This prevents accumulation and ensures only the **current documents** are used in retrieval
- I also implemented **automatic discarding of local ChromaDB files after each session**, so it doesn’t unnecessarily grow over time

I acknowledge that this approach will then limit only the currently uploaded documents. However, I discuss about this further in the **Future Work** section.

### Future Work & Smarter Retrieval

This fix works for now, but it sacrifices long-term memory.

#### Smarter Pipeline Idea:

Instead of hard-resetting the DB, we can **retrieve broadly**, then **let GPT semantically filter**:

1. Retrieve top-k (e.g., 20–30) chunks from **all documents** (no filter)
2. Inject them all into a single prompt
3. Ask GPT to:
   - Ignore irrelevant context
   - Answer only using the relevant parts
   - Optionally cite which chunks were used

This allows:

- **Mixed-source retrieval** (current + past)
- **Semantic filtering** inside the prompt, not just vector matching
- Better performance even when embeddings are imperfect or overlapping

---

## Problem 2: OpenAI Embeddings vs. BERT-Based Embeddings

### The Question

A key design decision I faced early on was:

> _"Should I use OpenAI's embedding models (like `text-embedding-ada-002`) or a BERT-based embedding approach like `Sentence-BERT` for my retrieval pipeline?"_

This decision greatly affects retrieval accuracy, performance, and overall system coherence.

### Research Reference

I referred to this in-depth comparison:  
[The Battle of Language Models: OpenAI vs. BERT](https://medium.com/@moradiyabhavik/the-battle-of-language-models-openai-vs-bert-ee46f4e5ef2f)

Key insights from the article:

- **OpenAI Embeddings** perform better on **general domain** tasks and have broader contextual understanding.
- **BERT (and Sentence-BERT)** embeddings, while slightly weaker in general contexts, often outperform in **specific domain-focused tasks** due to their structure-preserving embedding approach.
- OpenAI embeddings are paid and use external APIs, introducing latency and cost.

### My Implementation Decision

Based on this analysis and my own experimentation, I chose to use:

```python
self.embedding_function = HuggingFaceEmbeddings()
```

(See `src/core/quanta.py`)

Here's why:

- My project is **domain-specific** (mostly academic/research PDFs)
- Sentence-BERT showed **higher consistency** in retrieving contextually relevant chunks, especially for technical or scientific queries
- It avoids OpenAI API latency, quota issues, and cost - ideal for **local development and iteration**
- Embeddings are generated **locally**, giving me full control over performance and persistence (although I was not fully sure if this would make a difference significantly)

> Furthermore, the numerous testings I've performed at `src/testing` has given me confidence that BERT will work well for this product.

### ✅ How It's Used

In the `Quanta` class:

- Every document chunk is embedded using the Sentence-BERT encoder:
  ```python
  def embed(self, text):
      return self.embedding_function.embed_query(text)
  ```
- These embeddings are stored in ChromaDB and used for **cosine similarity-based retrieval**
- During querying, the user's query is also embedded using the same Sentence-BERT model to ensure **embedding consistency**

### Future Work

While BERT has worked well so far, I could consider hybrid options:

1. **Maybe... Multi-vector or Re-ranking Models?**
   - Use OpenAI or LLM to re-rank top-k chunks retrieved by Sentence-BERT

---

## Problem 3: From `map_reduce` Summarization to Contextual RAG

### The Original Approach: `map_reduce` Method Summarization

When I first started building the chatbot, my intuition was to use a **summarization-first pipeline**.

> In easy terms, I was thinking of making summary of summaries to deal with the massive amount of data (our intended purpose of making this chatbot).

So, I used a **`map_reduce` summarization strategy**:

- Split the document into chunks, calculating how much "tokens" I need.
- Generate summaries for each chunk ("map")
- Combine those summaries and summarize again ("reduce")

This method worked to a degree, but it **restricted the chatbot’s capabilities**:

- It only had access to a **summary**, not the actual document ultimately.
- Lost too many information at times, and it was hard to determind "how much" to summarize, as it differs based on the file size uploaded.
- If the summary missed a nuance, the bot couldn't retrieve it

---

### The Problem With That Approach

As mentioned above,

1. **Loss of Detail**: Important technical or nuanced points were lost in summarization layers.
2. **Static Summaries**: Every query used the same static summary, even if the query was specific.
3. **Poor Relevance for QA**: The system couldn't get into precise sections of the document.

---

### ✅ New Approach: Chunked Retrieval + GPT-4 Response (RAG)

Instead of generating summaries upfront, I moved to a **retrieval-based system**.

In my current architecture (see `quanta.py`):

- I **chunk the documents** using:

  ```python
  documents = DocumentProcessor.chunk_documents(batch_paths, self.document_store)
  ```

- Each chunk is embedded (using BERT) and stored in ChromaDB:

  ```python
  embeddings = [self.embed(doc.page_content) for doc in documents]
  self.document_store.add_documents(documents, embeddings=embeddings)
  ```

- During a query, I retrieve the most relevant chunks (cosine similarity):

  ```python
  results = self.document_store.similarity_search_by_vector(
      embedding=query_embedding,
      k=top_n_chunks,
  )
  ```

- Then, I **pass the retrieved chunks and query to GPT-4** for real-time reasoning:
  ```python
  def generate_response(self, query, chunks):
      # ... inject into prompt and call self.llm.invoke(prompt)
  ```

---

### Why New Approach Is More Effective

This architecture also deals with large files, but does so using **semantic retrieval + chunking**, which avoids the downsides of `map_reduce`. Here’s how:

#### 1. Chunking

Divided large documents into smaller, manageable chunks.

This avoids token overflows while **preserving original wording and structure** - unlike original summarization.

#### 2. Embedding + Vector Search

Each chunk is embedded and stored. During a query, **only the top-k relevant chunks** are retrieved.

#### 3. Dynamic Prompt Construction

Only those top-k chunks (based on cosine similarity to the query) are fed into the prompt:

```
Context: [Top-k Chunks]
User query: [User's question]
```

This keeps total token count **within GPT’s limits**, no matter how large the original file is.
