def evaluate_retrieval(evaluation_dataset, create_query_embeddings, retrieve_chunk, index, chunks, top_k=3, max_distance=0.9):

    results = []

    for item in evaluation_dataset:

        question = item["question"]
        expected_sources = item["expected_sources"]

        query_embedding = create_query_embeddings(question)

        retrieved_chunks = retrieve_chunk(query_embedding, index, chunks, top_k=top_k, max_distance=max_distance)
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

        if len(retrieved_sources) == 0:
            precision = 0
        else:
            precision = relavant_count / len(retrieved_sources)
        total_precision += precision

    average_precision = total_precision / len(results)

    return average_precision

def evaluate_answers(answer_evaluation_dataset, create_query_embeddings, retrieve_chunk, create_prompt, generate_answer, index, chunks, top_k=3, max_distance=0.9):
    results = []

    for item in answer_evaluation_dataset:
        question = item["question"]
        expected_answer = item["expected_answer"]

        query_embedding = create_query_embeddings(question)

        retrieved_chunks = retrieve_chunk(query_embedding, index, chunks, top_k=top_k, max_distance=max_distance)

        context = ""

        for chunk in retrieved_chunks:
            context += f"""
Source: {chunk["source"]}

{chunk["text"]}

"""
        prompt = create_prompt(context, question, [])
        answer = generate_answer(prompt)

        results.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": answer,
                "context": context
            }
        )

    return results

def judge_answer(question, expected_answer, generated_answer, generate_answer):
    prompt = f"""
You are an evaluator for a Retrieval-Augmented Generation system.

Evaluate the generated answer against the expected answer.

Question:
{question}

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Evaluate whether the generated answer correctly answers
the question and matches the important information in
the expected answer.

Give a score from 0 to 2:

0 = Incorrect or does not answer the question
1 = Partially correct or missing important information
2 = Correct and sufficiently complete

Return your response in exactly this format:

Score: <0, 1, or 2>
Reason: <short explanation>
"""
    
    evaluation = generate_answer(prompt)

    return evaluation

def judge_groundedness(
    question,
    context,
    generated_answer,
    generate_answer
):
    prompt = f"""
You are evaluating the groundedness of a RAG system.

Determine whether the generated answer is supported by
the retrieved context.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{generated_answer}

Give a score from 0 to 2:

0 = The answer contains information that is not supported
    by the retrieved context.

1 = The answer is partially supported, but contains some
    unsupported or questionable information.

2 = The answer is fully supported by the retrieved context
    and does not introduce unsupported factual claims.

Return your response in exactly this format:

Score: <0, 1, or 2>
Reason: <short explanation>
"""

    evaluation = generate_answer(prompt)

    return evaluation

def extract_score(evaluation):
    """
    Extract the numerical score from an LLM evaluation.

    Expected format:
    Score: 0
    Score: 1
    Score: 2
    """

    for line in evaluation.splitlines():

        if line.startswith("Score:"):
            score = line.split(":", 1)[1].strip()

        try:
            return int(score)
        except ValueError:
            return None

    return None

def calculate_answer_metrics(evaluations):
    """
    Calculate answer evaluation metrics.

    evaluations should be a list of LLM judge responses.
    """
    score_0 = 0
    score_1 = 0
    score_2 = 0

    for evaluation in evaluations:

        score = extract_score(evaluation)

        if score == 0:
            score_0 += 1

        elif score == 1:
            score_1 += 1

        elif score == 2:
            score_2 += 1

    total = score_0 + score_1 + score_2

    if total == 0:
        accuracy = 0
    else:
        accuracy = score_2 / total

    return {
        "score_0": score_0,
        "score_1": score_1,
        "score_2": score_2,
        "accuracy": accuracy
    }

def calculate_groundedness_metrics(evaluations):
    """
    Calculate groundedness evaluation metrics.

    evaluations should be a list of LLM groundedness
    judge responses.
    """

    score_0 = 0
    score_1 = 0
    score_2 = 0

    for evaluation in evaluations:

        score = extract_score(evaluation)

        if score == 0:
            score_0 += 1

        elif score == 1:
            score_1 += 1

        elif score == 2:
            score_2 += 1

    total = score_0 + score_1 + score_2

    if total == 0:
        groundedness_rate = 0
    else:
        groundedness_rate = score_2 / total

    return {
        "score_0": score_0,
        "score_1": score_1,
        "score_2": score_2,
        "groundedness_rate": groundedness_rate
    }