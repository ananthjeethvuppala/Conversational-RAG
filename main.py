from modules.pdf_loader import load_pdfs
from modules.chunker import chunks_documents
from modules.embedder import create_embeddings, create_query_embeddings
from modules.faiss_index import create_faiss_index
from modules.retriever import retrieve_chunk
from modules.prompts import create_prompt
from modules.llm import generate_answer

# --------------------------------------------------
# 1. Load PDF documents
# --------------------------------------------------

documents = load_pdfs("documents")
print(f"\nLoaded {len(documents)} PDF documents.")

# --------------------------------------------------
# 2. Create chunks
# --------------------------------------------------

chunks = chunks_documents(documents)
print(f"Created {len(chunks)} chunks.")

# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

embeddings = create_embeddings(chunks)
print(f"Embedding shape: {embeddings.shape}")

# --------------------------------------------------
# 4. Create FAISS index
# --------------------------------------------------

index = create_faiss_index(embeddings)
print(f"FAISS index contains {index.ntotal} vectors.")

# --------------------------------------------------
# 5. Get query from user
# --------------------------------------------------

print("\n" + "=" * 60)
print("MULTI-PDF RAG ASSISTANT")
print("=" * 60)

print("Ask questions about your PDF documents.")
print("Type 'exit' to quit.")

while True:

    query = input("\nYou: ").strip()

    if query.lower() in ["exit", "quit"]:
        print("\nExiting Multi-PDF RAG Assistant...")
        break

    if not query:
        print("Please enter a question.")
        continue

    # --------------------------------------------------
    # Create query embedding
    # --------------------------------------------------

    query_embedding = create_query_embeddings(query)

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

    prompt = create_prompt(context, query)

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    answer = generate_answer(prompt)

    # --------------------------------------------------
    # Display answer
    # --------------------------------------------------

    print("\nAssistant:")
    print(answer)

    sources = set()

    for result in results:
        sources.add(result["source"])

    print("\nSources:")
    
    for source in sources:
        print("-", source)