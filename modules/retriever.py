import numpy as np

def retrieve_chunk(query_embedding, index, chunks, top_k=3, max_distance=1.2):
    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_chunks = []

    for distance, index_position in zip(distances[0], indices[0]):

        if distance > max_distance:
            continue

        chunk = chunks[index_position]
        retrieved_chunks.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "distance": float(distance)
            }
        )

    return retrieved_chunks