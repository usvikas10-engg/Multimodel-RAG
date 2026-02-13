import faiss


class FAISSRetriever:
    def __init__(self, embeddings, metadata):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.metadata = metadata

    def search(self, query_embedding, top_k=5, filter_type=None):
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0:
                continue

            meta = self.metadata[idx]

            if filter_type and meta["type"] != filter_type:
                continue

            results.append(
                {
                    "id": int(idx),
                    "score": float(distances[0][rank]),
                    "metadata": meta
                }
            )

        return results
        
