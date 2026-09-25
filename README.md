# 🧠 Conversational RAG Assistant

A **Conversational Retrieval-Augmented Generation (RAG) system** that allows users to have multi-turn conversations with a collection of PDF documents.

The system combines **Multi-Document RAG with conversation memory and RAG evaluation**. It retrieves relevant information from multiple PDFs, uses conversation history to understand follow-up questions, rewrites contextual questions into standalone queries, generates grounded answers using an LLM through the **Groq API**, and evaluates retrieval quality, answer correctness, and groundedness.

---

## 🚀 Features

* 📄 **Multi-PDF Support**

  * Loads multiple PDF documents from a single directory.

* ✂️ **Document Chunking**

  * Splits documents into smaller overlapping chunks for efficient retrieval.

* 🧠 **Semantic Embeddings**

  * Uses Sentence Transformers with `all-MiniLM-L6-v2`.

* 🔎 **FAISS Semantic Search**

  * Retrieves the most relevant document chunks based on vector similarity.

* 🎯 **Distance-Based Retrieval Filtering**

  * Filters irrelevant chunks using a configurable maximum distance.

* 💬 **Conversation Memory**

  * Maintains previous user questions and assistant responses during the session.

* 🔄 **Question Rewriting**

  * Converts follow-up questions into standalone questions using conversation history.

* 🧩 **Context-Aware Retrieval**

  * Uses the rewritten question for semantic document retrieval.

* 🤖 **LLM Answer Generation**

  * Uses `openai/gpt-oss-120b` through the Groq API.

* 📚 **Source Attribution**

  * Displays the PDF documents used to generate each answer.

* 🔁 **Interactive Conversation**

  * Users can ask multiple questions without restarting the application.
  * Type `exit` or `quit` to end the session.

* 🧠 **Sliding-Window Memory**

  * Uses the most recent five conversation turns when providing history to the LLM.

* 📊 **RAG Evaluation**

  * Evaluates retrieval quality using Hit Rate@K and Precision@K.

* 🎯 **Answer Evaluation**

  * Uses an LLM judge to evaluate generated answers against expected answers.

* 🔍 **Groundedness Evaluation**

  * Evaluates whether generated answers are supported by the retrieved document context.

---

## 🏗️ Architecture

```text
                         PDF Documents
                              │
                              ▼
                       ┌──────────────┐
                       │ PDF Loader   │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │   Chunker    │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │  Embeddings  │
                       │ MiniLM-L6-v2 │
                       └──────┬───────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ FAISS Index  │
                       └──────┬───────┘
                              │
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
         Conversation History        Current Question
                │                           │
                └─────────────┬─────────────┘
                              ▼
                     Question Rewriter
                              │
                              ▼
                     Standalone Question
                              │
                              ▼
                       Query Embedding
                              │
                              ▼
                        FAISS Search
                              │
                              ▼
                    Retrieved PDF Chunks
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Conversation History │
                  │          +            │
                  │ Retrieved PDF Context│
                  └───────────┬───────────┘
                              │
                              ▼
                       Prompt Generator
                              │
                              ▼
                         Groq LLM
                              │
                              ▼
                       Generated Answer
                              │
                              ▼
                       Source Attribution
                              │
                              ▼
                    Store Conversation
                              │
                              └──────► Next Question
```

---

## 📁 Project Structure

```text
Conversational RAG/
│
├── documents/
│   ├── Artificial Intelligence.pdf
│   ├── Computer Vision.pdf
│   ├── Data Science.pdf
│   ├── Deep Learning.pdf
│   ├── Generative AI.pdf
│   ├── Large Language Models.pdf
│   ├── Machine Learning.pdf
│   ├── Natural Language Processing.pdf
│   ├── Python.pdf
│   ├── RAG.pdf
│   ├── SQL.pdf
│   └── Transformers.pdf
│
├── modules/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── faiss_index.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── llm.py
│   └── question_rewriter.py
│   ├── evaluation.py
│   ├── evaluation_dataset.py
│   └── answer_evaluation_dataset.py
│
├── main.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

| Technology                | Purpose                              |
| ------------------------- | ------------------------------------ |
| **Python**                | Core programming language            |
| **PyPDF**                 | PDF text extraction                  |
| **Sentence Transformers** | Semantic embeddings                  |
| **all-MiniLM-L6-v2**      | Embedding model                      |
| **FAISS**                 | Vector similarity search             |
| **NumPy**                 | Numerical operations                 |
| **Groq API**              | LLM inference and question rewriting |
| **GPT-OSS-120B**          | Answer generation                    |
| **python-dotenv**         | Environment variable management      |

---

## 🔄 How It Works

### 1. Load PDF Documents

PDF files placed inside the `documents/` directory are automatically loaded.

Each document retains its source filename:

```python
{
    "text": "...",
    "source": "Machine Learning.pdf"
}
```

This metadata is preserved throughout the retrieval pipeline.

---

### 2. Chunk the Documents

The extracted text is divided into smaller overlapping chunks.

Current configuration:

```text
Chunk size: 500
Overlap:    50
```

---

### 3. Generate Embeddings

Each document chunk is converted into a 384-dimensional vector using:

```text
all-MiniLM-L6-v2
```

These vectors represent the semantic meaning of the chunks.

---

### 4. Build the FAISS Index

The document embeddings are stored in a FAISS `IndexFlatL2` index.

This allows the system to search the entire PDF collection for chunks that are semantically related to a user's question.

---

### 5. Maintain Conversation Memory

Each interaction is stored as:

```python
{
    "user": "What is a transformer?",
    "assistant": "A transformer is..."
}
```

The conversation history is maintained during the current session.

---

### 6. Rewrite Follow-Up Questions

A follow-up question such as:

```text
Who introduced it?
```

may not contain enough information for semantic retrieval.

The question rewriter uses conversation history to convert it into a standalone query:

```text
Who introduced the transformer architecture?
```

The standalone query is then converted into an embedding and sent to FAISS.

---

### 7. Retrieve Relevant Context

The retriever searches the FAISS index and returns the most relevant chunks.

Current configuration:

```text
Top-K: 3
Maximum distance: 1.2
```

The distance threshold is experimental and can be adjusted based on evaluation results.

---

### 8. Combine Conversation + Retrieved Context

The final prompt contains:

```text
Conversation History
        +
Retrieved PDF Context
        +
Current Question
```

This allows the LLM to understand the current conversation while grounding its factual answer in the retrieved documents.

---

### 9. Generate the Answer

The prompt is sent to the Groq API using:

```text
openai/gpt-oss-120b
```

The generated answer is returned to the user.

---

### 10. Display Sources

The system extracts unique source filenames from the retrieved chunks.

Example:

```text
Sources:
- Transformers.pdf
- Generative AI.pdf
- Large Language Models.pdf
```

This provides basic traceability for the generated response.

---

## 📊 RAG Evaluation

The project includes an evaluation framework to measure the quality of the RAG pipeline.

The evaluation covers three areas:

### 1. Retrieval Evaluation

Retrieval quality is measured using:

- Hit Rate@K
- Precision@K
- Distance threshold experiments

### 2. Answer Evaluation

An LLM judge evaluates generated answers against expected answers using a score from 0 to 2:

| Score | Meaning |
|---|---|
| 0 | Incorrect or does not answer the question |
| 1 | Partially correct or missing important information |
| 2 | Correct and sufficiently complete |

### 3. Groundedness Evaluation

The generated answer is evaluated against the retrieved context:

| Score | Meaning |
|---|---|
| 0 | Answer contains unsupported information |
| 1 | Answer is partially supported |
| 2 | Answer is fully supported by the retrieved context |

### Retrieval Threshold Experiment

The retrieval distance threshold was evaluated using the retrieval evaluation dataset.

| Distance Threshold | Hit Rate@3 | Precision@3 |
|---:|---:|---:|
| 0.5 | 50.00% | 50.00% |
| 0.7 | 62.50% | 54.17% |
| 0.9 | 100.00% | 85.42% |
| 1.1 | 100.00% | 83.33% |
| 1.3 | 100.00% | 83.33% |
| 1.5 | 100.00% | 79.17% |

Based on this experiment, a distance threshold of **0.9** was selected for the current system.

> The `0.9` threshold is specific to this dataset and retrieval configuration and should not be considered a universal value.

### Final Evaluation Results

#### Retrieval Evaluation

- **Hit Rate@3:** 100.00%
- **Precision@3:** 85.42%

#### Answer Evaluation

- **Score 2:** 12/15
- **Score 1:** 0/15
- **Score 0:** 3/15
- **Answer Score-2 Rate:** 80.00%

#### Groundedness Evaluation

- **Score 2:** 13/15
- **Score 1:** 1/15
- **Score 0:** 1/15
- **Groundedness Rate:** 86.67%

### Evaluation Notes

The three answer-evaluation failures occurred on questions about:

- Embeddings in RAG
- Chunking in RAG
- Cosine similarity in semantic search

These results are influenced by the information available in the current PDF collection and retrieved context.

The answer and groundedness evaluations use an LLM as the judge, so small variations between runs are possible because the evaluator itself is probabilistic.

## 💬 Example Conversation

```text
============================================================
CONVERSATIONAL RAG ASSISTANT
============================================================

Ask questions about your PDF documents.
Type 'exit' to quit.

You: What is a transformer?

Rewritten Query:
What is a transformer model in machine learning?

Assistant:
A transformer is a deep-learning architecture designed to
process sequential data using attention mechanisms...

Sources:
- Transformers.pdf
- Large Language Models.pdf
- Generative AI.pdf


You: Who introduced it?

Rewritten Query:
Who introduced the transformer architecture?

Assistant:
The Transformer architecture was introduced in 2017 in the
paper "Attention Is All You Need."

Sources:
- Transformers.pdf
- Large Language Models.pdf


You: Why was it important?

Rewritten Query:
Why was the Transformer architecture important?

Assistant:
Transformers improved parallel processing, long-range
dependency modeling, and scalability compared with
recurrent architectures.

Sources:
- Transformers.pdf
```

---

## 🧠 Conversation Memory

The system uses a **sliding-window approach** when passing conversation history to the LLM.

```python
conversation_history[-5:]
```

This means the most recent five conversation turns are used as context.

The complete history can still exist in the Python list, but only the most recent five turns are provided to the LLM.

This helps prevent the prompt from continuously growing during long conversations.

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your API key.

Add the following to `.gitignore`:

```text
.env
__pycache__/
*.pyc
```

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project

```bash
cd "Conversational RAG"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

### 5. Add PDF documents

Place your PDF files inside:

```text
documents/
```

### 6. Run the application

```bash
python main.py
```

---

## 📊 Current Configuration

```text
PDF documents:       12
Chunk size:          500
Chunk overlap:       50
Embedding model:     all-MiniLM-L6-v2
Embedding dimension: 384
FAISS index:         IndexFlatL2
Top-K retrieval:     3
Distance threshold:  0.9
Memory window:       5 turns
LLM:                 openai/gpt-oss-120b
```

---

## 🧪 Tested Capabilities

The system has been tested with questions involving:

* Single-document retrieval
* Multi-document retrieval
* Follow-up questions
* Context-dependent pronouns such as `"it"`
* Topic switching
* Conversation history
* Question rewriting
* Source attribution

Example:

```text
What is overfitting?
        ↓
Why is it a problem?

What is a transformer?
        ↓
How is it different from RNNs?
```

The system successfully rewrites these follow-up questions into standalone queries before retrieval.

---

## ⚠️ Current Limitations

* FAISS index is rebuilt whenever the application starts.
* Conversation memory exists only for the current session.
* Memory is limited to a fixed five-turn context window.
* Retrieval uses a fixed `top_k` value.
* Distance threshold requires further evaluation.
* PDF extraction quality depends on the source PDF.
* Answer and groundedness evaluations use an LLM-based judge.
* LLM judge results can vary slightly between runs.
* The evaluation dataset is currently limited in size.
* Retrieval evaluation uses expected source documents rather than human relevance judgments.
* No automated experiment tracking or evaluation dashboard.
* No persistent conversation storage.
* Currently runs through a command-line interface.

---

## 🔮 Future Improvements

* [ ] Improved retrieval strategies
* [ ] Document-level ranking
* [ ] Persistent FAISS index
* [ ] Conversation summarization
* [ ] Long-term conversation memory
* [ ] Streamlit interface
* [ ] Better error handling
* [ ] Production deployment
* [ ] RAG monitoring and observability
* [ ] Automated experiment tracking
* [ ] Larger evaluation dataset

---

## 🎯 Learning Objectives

This project focuses on extending a Multi-Document RAG system into a **Conversational RAG system**.

Key concepts learned:

* Conversation memory
* Multi-turn conversations
* Context-dependent questions
* Query contextualization
* Follow-up question rewriting
* Sliding-window memory
* Context-aware retrieval
* Conversation context vs. document context
* Source attribution
* Grounded LLM generation
* Retrieval evaluation
* Hit Rate@K
* Precision@K
* Retrieval threshold tuning
* Answer correctness evaluation
* LLM-as-a-judge
* Groundedness evaluation
* Automated metric calculation
* RAG quality analysis

### Core Concept

> **Conversation memory provides context, while RAG provides knowledge.**

The conversation history helps understand what the user means, while the retrieved documents provide the factual information used to generate the answer.

---

## 👨‍💻 Author

**Ananth Jeeth Vuppala**

B.Tech in Electronics & Communication Engineering
Aspiring Software Engineer | AI/ML | NLP | LLMs

---

## ⭐ Project Goal

The goal of this project is to understand and implement **Conversational RAG from scratch**, building on the Multi-Document RAG pipeline without relying on high-level frameworks such as LangChain.

Each component is implemented separately to understand how conversational retrieval systems work internally.