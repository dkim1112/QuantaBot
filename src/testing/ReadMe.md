# Importance of Capturing Semantic Meaning

After reading this section, please also checkout the README files located in `src/testing/comparison` and `src/testing/add_langchain`.

## (1) TF-IDF (No Semantics) vs. Vector Embeddings

To illustrate the importance of capturing semantic meaning in text retrieval, I began by comparing two fundamentally different approaches: **TF-IDF** and **vector embeddings**.

## TF-IDF: Frequency-Based, No Semantics

**TF-IDF (Term Frequency-Inverse Document Frequency)** is a traditional method that represents text based on word frequency. It highlights words that are frequent in a given document but rare across the entire corpus, making it useful for keyword-based search.

However, **TF-IDF lacks the ability to understand the contextual or semantic relationships** between words. For example, it treats _"car"_ and _"automobile"_ as completely unrelated terms.

> **Limitation:** No understanding of synonymy, polysemy, or sentence meaning.

## Vector Embeddings: Contextual & Semantic Representation

In contrast, **vector embeddings** are designed to encode semantic information. These models map words, sentences, or documents into high-dimensional vectors where semantically similar items are located closer together in the vector space.

- Word embeddings (ex. Word2Vec, GloVe)
- Sentence/document embeddings (ex. BERT, SBERT)

To quantify this, I use cosine similarity, measuring cosine of the angle between two vectors.

## Cosine Similarity: Measuring Semantic Closeness

Cosine similarity ranges from -1 to 1, where 1 means the vectors point in the same direction (high similarity), 0 means orthogonality (no similarity), and -1 means opposite directions (completely dissimilar). This makes it especially well-suited for comparing vector embeddings in tasks like semantic search, clustering, and document retrieval.

```python
# Cosine similarity formula
cos_sim = (A · B) / (||A|| * ||B||)
```

By comparing TF-IDF and vector (semantic) embeddings in retrieval tasks, it becomes evident that embeddings offer significant advantages in understanding and retrieving semantically relevant information-even when the exact keywords are not shared.

## Understanding the BEIR Datasets

In evaluating the performance of retrieval methods, my testing primarily relies on a single dataset:

> NFCorpus from the [BEIR (Benchmarking Information Retrieval)](https://github.com/beir-cellar/beir?tab=readme-ov-file).

You may wonder: why choose **BEIR methodology** over other datasets for assessment?

The reason lies in its alignment with the goals of both semantic understanding and retrieval efficiency, which are essential for applications like chatbots and intelligent information systems.

**NFCorpus** is specifically designed for information retrieval tasks, where the objective is to retrieve relevant documents based on natural language queries. It consists of over 3,000 fact-based documents, primarily drawn from the biomedical domain. Each query is paired with a set of relevance judgments, enabling the use of performance metrics such as F1 score, which requires ground truth labels for accurate evaluation. This makes NFCorpus an ideal benchmark for assessing how well a system understands and retrieves meaningful information.

Strong performance on this dataset suggests that the retrieval model is capable of handling complex, real-world queries - translating to more accurate and contextually relevant responses in chatbot applications (i.e. QuantaBot). Moreover, BEIR is known to provide a more standardized format (including documents, queries, and relevance labels), supporting reproducibility and fair comparison across retrieval methods.

> ℹ️ **Note**  
> Considering compile time, common limit has been set to 500 documents and 20 queries (can be easily adjusted inside the code).
