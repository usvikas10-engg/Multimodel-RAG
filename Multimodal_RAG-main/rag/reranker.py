def simple_rerank(query, docs):
    scored = []

    for doc in docs:
        score = sum(1 for w in query.lower().split() if w in doc.lower())
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in scored]


def simple_rerank_with_scores(query, docs):
    scored = []

    query_terms = [w for w in query.lower().split() if w]

    for i, doc in enumerate(docs):
        lexical_score = sum(1 for w in query_terms if w in doc.lower())
        scored.append(
            {
                "doc_id": i,
                "lexical_score": lexical_score,
                "doc": doc
            }
        )

    scored.sort(reverse=True, key=lambda x: x["lexical_score"])
    return scored
