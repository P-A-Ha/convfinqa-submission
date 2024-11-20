import logging

from dataloaders import ConfigLoad, QAdataloader
from front_end import GradioInterface
from model_inferences import EmbeddingClass, InstructClass
from rag import RAGComponents
from vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info(f"Loading Configurarion from Config")
    config_loader = ConfigLoad()
    config = config_loader.config

    logger.info(f"Loading Data from Files and Preparing Documents")
    data_loader = QAdataloader(config)
    qa_pairs = data_loader.load_qa()
    additional_context_texts = data_loader.load_additional_context()

    logger.info(f"Initiialising Embedding Model")
    embedding_model = EmbeddingClass(config["model"]["embedding_model_hf_name"])

    logger.info(f"Initiialising Vector Store Managing Module")
    vector_store_manager = VectorStoreManager(config, embedding_model.embeddings_model)
    vector_store_manager.create_qa_vector_store(qa_pairs)
    vector_store_manager.create_context_vector_store(additional_context_texts)

    logger.info(f"Initiialising Instruct Model")
    instruct_class = InstructClass(config)

    logger.info(f"Initiialising RAG")
    rag_system = RAGComponents(instruct_class, vector_store_manager)

    logger.info(f"Launching Gradio Interface")
    rag_interface = GradioInterface(instruct_class, rag_system, config)
    rag_interface.launch()


if __name__ == "__main__":
    main()
