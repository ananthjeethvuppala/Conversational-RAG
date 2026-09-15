from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    texts = []

    for chunk in chunks:
        texts.append(chunk["text"])

    embeddings = model.encode(texts)

    return embeddings

def create_query_embeddings(query):
    return model.encode(query)