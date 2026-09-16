from modules.pdf_loader import load_pdfs
from modules.chunker import chunks_documents
from modules.embedder import create_embeddings, create_query_embeddings
from modules.faiss_index import create_faiss_index
from modules.retriever import retrieve_chunk
from modules.prompts import create_prompt
from modules.llm import generate_answer
from modules.question_rewriter import rewrite_question
from modules.evaluation_dataset import evaluation_dataset
from modules.evaluation import evaluate_retrieval, calculate_hit_rate, calculate_precision_at_k

# --------------------------------------------------
# 1. Load PDF documents
# --------------------------------------------------

print("Loading PDFs...")
documents = load_pdfs("documents")
print(f"\nLoaded {len(documents)} PDF documents.\n")

# --------------------------------------------------
# 2. Create chunks
# --------------------------------------------------

print("Creating chunks...")
chunks = chunks_documents(documents)
print(f"Created {len(chunks)} chunks.\n")

# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

print("Creating embeddings...")
embeddings = create_embeddings(chunks)
print(f"Embedding shape: {embeddings.shape}\n")

# --------------------------------------------------
# 4. Create FAISS index
# --------------------------------------------------

print("Creating FAISS index...")
index = create_faiss_index(embeddings)
print(f"FAISS index contains {index.ntotal} vectors.")

# --------------------------------------------------
# 5. Evaluation
# --------------------------------------------------
for threshold in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:

    evaluation_results = evaluate_retrieval(evaluation_dataset, create_query_embeddings, retrieve_chunk, index, chunks, top_k=3, max_distance=threshold)

    hit_rate = calculate_hit_rate(evaluation_results)

    precision = calculate_precision_at_k(evaluation_results, k=3)

    # print("\nRetrieval Evaluation")
    # print("--------------------")

    # for result in evaluation_results:
    #     print(f"\nQuestion: {result['question']}")
    #     print(f"Expected: {result['expected_sources']}")
    #     print(f"Retrieved: {result['retrieved_sources']}")
    #     print(f"Hit: {result['hit']}")

    print(f"\nThreshold = {threshold}")
    print(f"\nHit Rate@3: {hit_rate * 100:.2f}%")
    print(f"Precision@3: {precision * 100:.2f}%")

# --------------------------------------------------
# 6. Get query from user
# --------------------------------------------------

print("\n" + "=" * 60)
print("CONVERSATIONAL RAG ASSISTANT")
print("=" * 60)

print("Ask questions about your PDF documents.")
print("Type 'exit' to quit.")

conversation_history = []

while True:

    query = input("\nYou: ").strip()

    if query.lower() in ["exit", "quit"]:
        print("\nExiting Conversational RAG Assistant...")
        break

    if not query:
        print("Please enter a question.")
        continue

    # --------------------------------------------------
    # Create query embedding
    # --------------------------------------------------

    standalone_query = rewrite_question(query, conversation_history)

    print("\nRewritten Query:")
    print(standalone_query)

    query_embedding = create_query_embeddings(standalone_query)

    # --------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------

    results = retrieve_chunk(
        query_embedding,
        index,
        chunks,
        top_k=3
    )

    # --------------------------------------------------
    # Check if anything was retrieved
    # --------------------------------------------------

    if not results:
        print("\nAssistant: I could not find relevant information in the provided documnets.")
        continue

    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context = ""

    for result in results:
        context += f"""
    Source: {result["source"]}

    {result["text"]}

    ---------------------------------------------------------------------
    """

    # --------------------------------------------------
    # Create prompt
    # --------------------------------------------------

    prompt = create_prompt(context, query, conversation_history)

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    answer = generate_answer(prompt)

    # --------------------------------------------------
    # Display answer
    # --------------------------------------------------

    print("\nAssistant:")
    print(answer)

    conversation_history.append(
        {
            "user": query,
            "assistant": answer
        }
    )

    sources = set()

    for result in results:
        sources.add(result["source"])

    print("\nSources:")
    
    for source in sources:
        print("-", source)

# --------------------------------------------------
# Un-comment the below for Conversation Memory
# --------------------------------------------------

# print("\nConversation Memory:")
# for conversation in conversation_history:
#     print(f"User: {conversation['user']}")
#     print(f"Assistant: {conversation['assistant']}")
#     print("-" * 40)