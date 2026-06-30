import numpy as np
import faiss
from config import EMBEDDING_DIM, FAISS_TOP_K


class CandidateIndex:
    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product = cosine on normalized vecs
        self.candidates: list[dict] = []

    def add_candidates(self, candidates: list[dict], embeddings: np.ndarray) -> None:
        self.candidates.extend(candidates)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = FAISS_TOP_K) -> list[tuple[dict, float]]:
        if self.index.ntotal == 0:
            return []
        query = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.candidates[idx], float(score)))
        return results

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.candidates = []
