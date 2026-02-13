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
                "doc": doc,
            }
        )

    scored.sort(reverse=True, key=lambda x: x["lexical_score"])
    return scored


def hybrid_rerank_with_scores(
    query,
    docs,
    distances=None,
    semantic_weight=0.7,
    lexical_weight=0.3,
):
    if distances is None:
        distances = [0.0] * len(docs)

    if len(distances) != len(docs):
        raise ValueError("distances length must match docs length")

    query_terms = [w for w in query.lower().split() if w]

    lexical_scores = []
    for doc in docs:
        lexical_scores.append(sum(1 for w in query_terms if w in doc.lower()))

    max_lex = max(lexical_scores) if lexical_scores else 1
    if max_lex == 0:
        max_lex = 1

    scored = []
    for i, doc in enumerate(docs):
        lexical_score = lexical_scores[i]
        lexical_norm = lexical_score / max_lex
        semantic_score = 1.0 / (1.0 + float(distances[i]))
        hybrid_score = (semantic_weight * semantic_score) + (lexical_weight * lexical_norm)

        scored.append(
            {
                "doc_id": i,
                "lexical_score": lexical_score,
                "semantic_score": semantic_score,
                "hybrid_score": hybrid_score,
                "doc": doc,
            }
        )

    scored.sort(reverse=True, key=lambda x: x["hybrid_score"])
    return scored
