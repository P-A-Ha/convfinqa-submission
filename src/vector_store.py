import logging

from langchain_milvus import Milvus
from pymilvus import Collection, connections, utility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self, config, embeddings_model):
        self.embeddings_model = embeddings_model
        self.host = config["vector_db"]["host_url"]
        self.port = config["vector_db"]["port_number"]
        self.qa_collection_name = config["vector_db"]["fin_qa_collection_name"]
        self.context_collection_name = config["vector_db"][
            "extra_fin_context_collection_name"
        ]
        self.connect()

    def connect(self):
        connections.connect(alias="default", host=self.host, port=self.port)

    def load_or_create_vectore_store(self, collection_name, documents):
        if utility.has_collection(collection_name):
            logger.info(f"Using existing collection: {collection_name}")
            collection = Milvus(
                collection_name=collection_name,
                embedding_function=self.embeddings_model,
            )
        else:
            logger.info(f"Creating new collection: {collection_name}")
            collection = Milvus.from_documents(
                documents,
                self.embeddings_model,
                collection_name=collection_name,
                drop_old=False,
            )
        return collection

    def create_qa_vector_store(self, qa_documents):
        self.qa_vector_store = self.load_or_create_vectore_store(
            self.qa_collection_name, qa_documents
        )

    def create_context_vector_store(self, context_documents):
        self.meta_vector_store = self.load_or_create_vectore_store(
            self.context_collection_name, context_documents
        )

    def get_qa_retriever(self):
        return self.qa_vector_store.as_retriever(search_kwargs={"k": 4})

    def get_meta_retriever(self):
        return self.meta_vector_store.as_retriever(search_kwargs={"k": 4})
