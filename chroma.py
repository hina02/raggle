import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# def from_documents(
#     cls: Type[Chroma],
#     documents: List[Document],
#     embedding: Optional[Embeddings] = None,
#     ids: Optional[List[str]] = None,
#     collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
#     persist_directory: Optional[str] = None,
#     client_settings: Optional[chromadb.config.Settings] = None,
#     client: Optional[chromadb.ClientAPI] = None,  # Add this line
#     collection_metadata: Optional[Dict] = None,
#     **kwargs: Any,
# ) -> Chroma:
class ChromaManager:
    def __init__(self, collection_name: str):
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        ids = self.vector_store.add_documents(documents)
        print(f"documents added to chroma {len(ids)} / {len(documents)}.")
        return ids

    def query(self, query: str, filter: dict, k: int = 4) -> list[tuple[Document, float]]:
        results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
            embedding=embeddings.embed_query(query),
            k=k,
            filter=filter,
            # where_document={"$contains": {"text": "hello"}},
        )
        # results = self.vector_store.max_marginal_relevance_search_by_vector
        return results
