import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.dataloaders import ConfigLoad, QAdataloader
from src.model_inferences import EmbeddingClass
from src.rag import RAGComponents
from src.vector_store import VectorStoreManager


class BatchEvaluation:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            config["model"]["embedding_model_hf_name"]
        )

    def compute_similarity(self, query, retrieved):
        embeddings = self.embedding_model.encode([query, retrieved])
        similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity_score)


if __name__ == "__main__":
    config_loader = ConfigLoad()
    config = config_loader.config
    data_loader = QAdataloader(config)
    qa_pairs = data_loader.load_qa()
    additional_context_texts = data_loader.load_additional_context()
    embedding_model = EmbeddingClass(config["model"]["embedding_model_hf_name"])
    vector_store_manager = VectorStoreManager(config, embedding_model.embeddings_model)
    vector_store_manager.create_qa_vector_store(qa_pairs)
    vector_store_manager.create_context_vector_store(additional_context_texts)
    rag_system = RAGComponents(
        instruct_model=None,
        vector_store_mngr=vector_store_manager,
        batch_eval_retrieval=True,
    )
    evaluation_class = BatchEvaluation()

    path = "testing/augmented_qa_data.json"

    with open(path, "r") as file:
        data = json.load(file)

    for item in data:
        synonym_replacement = item["synonym_replacement"]
        random_insertion = item["random_insertion"]
        random_swap = item["random_swap"]
        random_deletion = item["random_deletion"]
        random_complex = item["random_complex"]
        question = item["question"]

        retrieved_question, retrieved_answer = rag_system.retrieval_qa(question)
        item["mirror_score"] = evaluation_class.compute_similarity(
            retrieved_question, question
        )
        item["mirror_answer"] = retrieved_answer
        item["mirror_question"] = retrieved_question

        retrieved_question, retrieved_answer = rag_system.retrieval_qa(
            synonym_replacement
        )
        item["synonym_replacement_score"] = evaluation_class.compute_similarity(
            retrieved_question, synonym_replacement
        )
        item["synonym_replacement_answer"] = retrieved_answer

        retrieved_question, retrieved_answer = rag_system.retrieval_qa(random_insertion)
        item["random_insertion_score"] = evaluation_class.compute_similarity(
            retrieved_question, random_insertion
        )
        item["random_insertion_answer"] = retrieved_answer

        retrieved_question, retrieved_answer = rag_system.retrieval_qa(random_swap)
        item["random_swap_score"] = evaluation_class.compute_similarity(
            retrieved_question, random_swap
        )
        item["random_swap_answer"] = retrieved_answer

        retrieved_question, retrieved_answer = rag_system.retrieval_qa(random_deletion)
        item["random_deletion_score"] = evaluation_class.compute_similarity(
            retrieved_question, random_deletion
        )
        item["random_deletion_answer"] = retrieved_answer

        retrieved_question, retrieved_answer = rag_system.retrieval_qa(random_complex)
        item["random_complex_score"] = evaluation_class.compute_similarity(
            retrieved_question, random_complex
        )
        item["random_complex_answer"] = retrieved_answer

    with open("testing/eval-augmentation-final-2.json", "w") as file:
        updated_json = json.dump(data, file, indent=4)
