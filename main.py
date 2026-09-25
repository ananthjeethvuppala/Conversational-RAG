from modules.pdf_loader import load_pdfs
from modules.chunker import chunks_documents
from modules.embedder import create_embeddings, create_query_embeddings
from modules.faiss_index import create_faiss_index
from modules.retriever import retrieve_chunk
from modules.prompts import create_prompt
from modules.llm import generate_answer
from modules.question_rewriter import rewrite_question
from modules.evaluation_dataset import evaluation_dataset
from modules.evaluation import evaluate_retrieval, calculate_hit_rate, calculate_precision_at_k, evaluate_answers, judge_answer, judge_groundedness, extract_score, calculate_answer_metrics, calculate_groundedness_metrics
from modules.answer_evaluation_dataset import answer_evaluation_dataset

# --------------------------------------------------
# 1. Load PDF documents
# --------------------------------------------------

print("\nLoading PDFs...")
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
# 5. Evaluation and Chat Menu
# --------------------------------------------------

conversation_history = []

while True:

    print("\n" + "=" * 60)
    print("CONVERSATIONAL RAG ASSISTANT")
    print("=" * 60)

    print("\n1. Chat with RAG")
    print("2. Retrieval Evaluation")
    print("3. Answer Evaluation")
    print("4. Groundedness Evaluation")
    print("5. Exit")

    choice = input("\nSelect an option: ").strip()

    # --------------------------------------------------
    # 1. Chat with RAG
    # --------------------------------------------------

    if choice == "1":

        print("\n" + "=" * 60)
        print("CHAT WITH RAG")
        print("=" * 60)

        print("Ask questions about your PDF documents.")
        print("Type 'exit' to return to the main menu.")

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
    # 2. Retrieval Evaluation
    # --------------------------------------------------

    elif choice == "2":

        print("\n" + "=" * 60)
        print("RETRIEVAL EVALUATION")
        print("=" * 60)

        print("\nNumber of retrieval evaluation questions:",len(evaluation_dataset))

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
    # 3. Answer Evaluation
    # --------------------------------------------------

    elif choice == "3":

        print("\n" + "=" * 60)
        print("ANSWER EVALUATION")
        print("=" * 60)

        answer_results = evaluate_answers(answer_evaluation_dataset, create_query_embeddings, retrieve_chunk, create_prompt, generate_answer, index, chunks, top_k=3, max_distance=0.9)

        evaluations = []
        for result in answer_results:

            evaluation = judge_answer(result["question"], result["expected_answer"], result["generated_answer"], generate_answer)

            evaluations.append(evaluation)

            print(f"\nQuestion: {result['question']}")
            print(f"Expected Answer: {result['expected_answer']}")
            print(f"Generated Answer: {result['generated_answer']}")
            print(f"Evaluation: {evaluation}")

            print("\n" + "=" * 60)

        # -----------------------------------------
        # Final Answer Metrics
        # -----------------------------------------
        
        print("\nANSWER METRICS")
        print("--------------")

        metrics = calculate_answer_metrics(evaluations)

        print(f"Score 2: {metrics['score_2']}")
        print(f"Score 1: {metrics['score_1']}")            
        print(f"Score 0: {metrics['score_0']}")

        print(f"Answer Accuracy: {metrics['accuracy'] * 100:.2f}%")

    # --------------------------------------------------
    # 4. Groundedness Evaluation
    # --------------------------------------------------

    elif choice == "4":

        print("\n" + "=" * 60)
        print("GROUNDEDNESS EVALUATION")
        print("=" * 60)

        answer_results = evaluate_answers(answer_evaluation_dataset, create_query_embeddings, retrieve_chunk, create_prompt, generate_answer, index, chunks, top_k=3, max_distance=0.9)

        evaluations = []

        for result in answer_results:
            evaluation = judge_groundedness(
                result["question"],
                result["context"],
                result["generated_answer"],
                generate_answer
            )

            evaluations.append(evaluation)

            print(f"\nQuestion: {result['question']}")
            print(f"Generated Answer: {result['generated_answer']}")
            print(f"Groundedness: {evaluation}")
            print("\n" + "=" * 60)

        # -----------------------------------------
        # Final Groundedness Metrics
        # -----------------------------------------

        metrics = calculate_groundedness_metrics(
            evaluations
        )

        print("\nGROUNDEDNESS METRICS")
        print("--------------------")

        print(f"Score 2: {metrics['score_2']}")
        print(f"Score 1: {metrics['score_1']}")
        print(f"Score 0: {metrics['score_0']}")

        print(
            f"Groundedness Rate: "
            f"{metrics['groundedness_rate'] * 100:.2f}%")

    # --------------------------------------------------
    # 5. Exit
    # --------------------------------------------------

    elif choice == "5":

        print("\nExiting Conversational RAG Assistant...")
        break

    else:

        print("\nInvalid option. Please select 1-6.")

# --------------------------------------------------
# Un-comment the below for Conversation Memory
# --------------------------------------------------

# print("\nConversation Memory:")
# for conversation in conversation_history:
#     print(f"User: {conversation['user']}")
#     print(f"Assistant: {conversation['assistant']}")
#     print("-" * 40)