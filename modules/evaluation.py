def evaluate_retrieval(evaluation_dataset, create_query_embeddings, retrieve_chunk, index, chunks, top_k=3):

    results = []

    for item in evaluation_dataset:

        question = item["question"]
        expected_sources = item["expected_sources"]

        query_embedding = create_query_embeddings(question)

        retrieved_chunks = retrieve_chunk(query_embedding, index, chunks, top_k=top_k)
        retrieved_sources = [chunk["source"] for chunk in retrieved_chunks]

        hit = any(source in expected_sources for source in retrieved_sources)

        results.append(
            {
                "question": question,
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources,
                "hit": hit
            }
        )

    return results

def calculate_hit_rate(results):

    hits = 0
    for result in results:
        if result["hit"]:
            hits += 1

    hit_rate = hits / len(results)

    return hit_rate

def calculate_precision_at_k(results, k=3):

    total_precision = 0

    for result in results:

        expected_sources = result["expected_sources"]
        retrieved_sources = result["retrieved_sources"][:k]

        relavant_count = 0

        for source in retrieved_sources:
            if source in expected_sources:
                relavant_count += 1

        precision = relavant_count / len(retrieved_sources)
        total_precision += precision

    average_precision = total_precision / len(results)

    return average_precision