from __future__ import annotations

from typing import Sequence

from chromadb import PersistentClient
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


def ensure_embedding(model_name: str) -> None:
    # Configure the global embed model once.
    if not isinstance(getattr(Settings, "embed_model", None), HuggingFaceEmbedding):
        Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)


def get_chroma_vector_store(persist_dir: str, collection: str = "docs") -> ChromaVectorStore:
    client = PersistentClient(path=persist_dir)
    col = client.get_or_create_collection(name=collection)
    return ChromaVectorStore(chroma_collection=col)


def index_nodes(
    nodes: Sequence[BaseNode], persist_dir: str, embedding_model: str, collection: str = "docs"
) -> VectorStoreIndex:
    ensure_embedding(embedding_model)
    vector_store = get_chroma_vector_store(persist_dir, collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # Build an index on top of the vector store
    return VectorStoreIndex(nodes, storage_context=storage_context)


def load_index(
    persist_dir: str, embedding_model: str, collection: str = "docs"
) -> VectorStoreIndex:
    ensure_embedding(embedding_model)
    vector_store = get_chroma_vector_store(persist_dir, collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )
