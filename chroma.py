import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document


class ChromaManager:
    def __init__(self, client, collection_name):

        persistent_client = chromadb.PersistentClient()
        collection = persistent_client.get_or_create_collection("collection_name")
        collection.add(ids=["1", "2", "3"], documents=["a", "b", "c"])

        vector_store_from_client = Chroma(
            client=persistent_client,
            collection_name="collection_name",
            embedding_function=embeddings,
        )

    def add_documents(self, documents):
        pass

    def query(self, query):
        pass
    