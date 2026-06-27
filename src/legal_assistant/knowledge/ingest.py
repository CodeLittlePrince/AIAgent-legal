"""法律 Markdown 文档向量化入库流程。

读取 ``profile/legal`` 目录下的法条 Markdown，分块、嵌入后写入 Chroma 向量库，
供 ``LegalRetriever`` 检索使用。
"""

from pathlib import Path

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from legal_assistant.config import settings

# Chroma 中法律知识集合的名称（与 retriever 保持一致）
LEGAL_COLLECTION = "legal_knowledge"
# 默认法条文档目录：项目根目录下的 profile/legal
PROFILE_LEGAL_DIR = Path(__file__).resolve().parents[3] / "profile" / "legal"


def _get_embedding():
    """创建 HuggingFace 嵌入模型实例，模型名称来自配置。"""
    return HuggingFaceEmbedding(model_name=settings.embedding_model)


def _get_chroma_client(chroma_host: str | None = None, chroma_port: int | None = None):
    """根据主机地址选择 Chroma 客户端类型。

    本地开发（localhost / 127.0.0.1）使用项目内 ``.chroma_data`` 持久化目录；
    其他地址使用 HTTP 客户端连接远程 Chroma 服务。
    """
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
    """将 Markdown 法条文档解析、分块并写入 Chroma 向量库。

    流程概要：
    1. 读取指定目录下所有 ``.md`` 文件
    2. 用 MarkdownNodeParser 切分为节点（chunk）
    3. 删除旧集合并重建，避免重复入库产生脏数据
    4. 通过 LlamaIndex VectorStoreIndex 计算嵌入并写入 Chroma

    Args:
        chroma_host: Chroma 服务主机，可选。
        chroma_port: Chroma 服务端口，可选。
        profile_dir: 法条 Markdown 根目录，默认 ``profile/legal``。

    Returns:
        成功索引的节点（chunk）数量；无文档时返回 0。
    """
    docs_dir = profile_dir or PROFILE_LEGAL_DIR
    embed_model = _get_embedding()
    Settings.embed_model = embed_model  # 设置 LlamaIndex 全局默认嵌入模型

    reader = SimpleDirectoryReader(
        input_dir=str(docs_dir),
        required_exts=[".md"],
        exclude_hidden=True,
        filename_as_id=True,  # 用文件名作为文档 ID，便于溯源
    )
    documents = reader.load_data()
    if not documents:
        return 0

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    client = _get_chroma_client(chroma_host, chroma_port)
    # 全量重建：先删旧集合，忽略「集合不存在」类异常
    try:
        client.delete_collection(LEGAL_COLLECTION)
    except Exception:
        pass
    chroma_collection = client.get_or_create_collection(LEGAL_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # 构建索引时会自动对 nodes 做嵌入并写入 vector_store
    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    return len(nodes)
