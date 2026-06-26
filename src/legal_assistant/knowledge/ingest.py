from pathlib import Path

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from legal_assistant.config import settings

LEGAL_COLLECTION = "legal_knowledge"
PROFILE_LEGAL_DIR = Path(__file__).resolve().parents[3] / "profile" / "legal"


def _get_embedding():
    return HuggingFaceEmbedding(model_name=settings.embedding_model)


def _get_chroma_client(chroma_host: str | None = None, chroma_port: int | None = None):
    host = chroma_host or settings.chroma_host
    port = chroma_port or settings.chroma_port
    if host in ("localhost", "127.0.0.1"):
        return chromadb.PersistentClient(path=str(Path(".chroma_data")))
    return chromadb.HttpClient(host=host, port=port)


def ingest_legal_documents(
    chroma_host: str | None = None,
    chroma_port: int | None = None,
    profile_dir: Path | None = None,
) -> int:
    """Ingest markdown legal docs into Chroma. Returns number of nodes indexed."""
    docs_dir = profile_dir or PROFILE_LEGAL_DIR
    embed_model = _get_embedding()
    Settings.embed_model = embed_model

    reader = SimpleDirectoryReader(
        input_dir=str(docs_dir),
        required_exts=[".md"],
        exclude_hidden=True,
        filename_as_id=True,
    )
    documents = reader.load_data()
    if not documents:
        return 0

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    client = _get_chroma_client(chroma_host, chroma_port)
    try:
        client.delete_collection(LEGAL_COLLECTION)
    except Exception:
        pass
    chroma_collection = client.get_or_create_collection(LEGAL_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    return len(nodes)
