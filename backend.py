import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')


def load_faiss():
    """
    Loads the FAISS index and associated metadata from disk.
    Returns:
        index (faiss.Index): FAISS search index
        metadata (list): List of metadata items corresponding to the index
    """
    try:
        index = faiss.read_index("faiss_manual_index/index.faiss")
        with open("faiss_manual_index/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index or metadata: {e}")


def embed_query(query: str) -> np.ndarray:
    """
    Converts a user query into a dense vector using a SentenceTransformer model.

    Args:
        query (str): The input text query

    Returns:
        np.ndarray: Embedded query vector of shape (1, embedding_dim)
    """
    embedding = model.encode([query])  
    return np.array(embedding).astype('float32')


def search_faiss(index, query_vector: np.ndarray, top_k: int = 5):
    """
    Searches the FAISS index with the embedded query vector.

    Args:
        index (faiss.Index): The FAISS index
        query_vector (np.ndarray): Embedded query vector
        top_k (int): Number of top results to return

    Returns:
        distances (np.ndarray): Distances of top matches
        indices (np.ndarray): Indices of top matches
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return distances, indices


def get_results(indices: np.ndarray, metadata: list) -> list:
    """
    Retrieves matching results from metadata based on FAISS indices.

    Args:
        indices (np.ndarray): FAISS result indices
        metadata (list): Original metadata entries

    Returns:
        list: Retrieved metadata entries
    """
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            item=metadata[idx]
            if isinstance(item, dict):
                results.append(item.get("text") or str(item))
            else:
                results.append(str(item))
        else:
            results.append("⚠️ Metadata not found for index: " + str(idx))
    return results
