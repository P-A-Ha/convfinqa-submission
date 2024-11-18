from ragas.llms import LangchainLLMWrapper

from dataloaders import ConfigLoad, QAdataloader
from model_inferences import EmbeddingClass, InstructClass
from rag import RAGComponents
from vector_store import VectorStoreManager


class BatchEvaluation:
    def __init__(self, llm_model):
        self.llm_model = llm_model

    def batch_eval(self):
        evaluation = "Response meets the quality standards."
        return evaluation


if __name__ == "__main__":
    config_loader = ConfigLoad()
    config = config_loader.config
    llm_model = InstructClass(config)
    data_loader = QAdataloader(config)
    qa_pairs = data_loader.load_qa()
    embedding_model = EmbeddingClass(config["model"]["embedding_model_hf_name"])
    vector_store_manager = VectorStoreManager(config, embedding_model.embeddings_model)
    vector_store_manager.create_qa_vector_store(qa_pairs)
    llm_model = InstructClass(config)
    rag_system = RAGComponents(llm_model, vector_store_manager)
    BatchEvaluation(llm_model, eval_document)
