def chunks_documents(documents, chunk_size=500, chunk_overlap=50):
    all_chunks = []
    step = chunk_size - chunk_overlap

    for document in documents:
        text = document["text"]
        source = document["source"]

        for index in range(0, len(text), step):
            chunk = text[index : index + chunk_size]
            if chunk.strip():
                all_chunks.append(
                    {
                        "text" : chunk.strip(),
                        "source" : source
                    }
                )
    return all_chunks