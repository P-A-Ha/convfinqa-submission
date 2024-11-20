import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from dataloaders import ConfigLoad

config_loader = ConfigLoad()
config = config_loader.config
nltk.download("punkt_tab")


class RAGEvaluation:
    def __init__(
        self,
        retrieved_question,
        corresponding_qa_answer,
        generative_response,
        query,
        instruct_model,
        additional_context_text=None,
    ):
        self.retrieved_question = retrieved_question
        self.corresponding_answer = corresponding_qa_answer
        self.generative_response = generative_response
        self.additional_context_text = additional_context_text
        self.query = query
        self.instruct_model = instruct_model

        self.embedding_model = SentenceTransformer(
            config["model"]["embedding_model_hf_name"]
        )
        self.rouge_calculator = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.smoothing_function = SmoothingFunction()

    def compute_similarity(self, query, retrieved):
        embeddings = self.embedding_model.encode([query, retrieved])
        similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity_score)

    def compute_bleu(self, reference, generated):
        reference_tokens = nltk.word_tokenize(reference.lower())
        generated_tokens = nltk.word_tokenize(generated.lower())
        bleu_score = sentence_bleu(
            [reference_tokens],
            generated_tokens,
            smoothing_function=self.smoothing_function.method1,
        )
        return bleu_score

    def compute_rouge(self, reference, generated):
        rouge_scores = self.rouge_calculator.score(reference, generated)
        return rouge_scores

    def output_evaluation(self):
        evaluation = {}

        if self.additional_context_text is None:
            # Retrieval Similarity to Query
            similarity_score_retrieval = self.compute_similarity(
                self.query, self.retrieved_question
            )
            evaluation["Retrieved_Question_Similarity_to_User_Query"] = similarity_score_retrieval

            rouge_scores = self.compute_rouge(
                self.corresponding_answer, self.generative_response
            )
            evaluation["Rouge1_RetrievedAnswer_vs_Generation"] = rouge_scores["rouge1"].fmeasure
            evaluation["Rouge2_RetrievedAnswer_vs_Generation"] = rouge_scores["rouge2"].fmeasure
            evaluation["RougeL_RetrievedAnswer_vs_Generation"] = rouge_scores["rougeL"].fmeasure

            bleu_score = self.compute_bleu(
                self.corresponding_answer, self.generative_response
            )
            evaluation["BLEU_score_RetrievedAnswer_vs_Generation"] = bleu_score

        else:
            # Retrieval Similarity to Query
            similarity_score_retrieval = self.compute_similarity(
                self.query, self.additional_context_text
            )
            evaluation["Retrieved_Context_Similarity_to_User_Query"] = similarity_score_retrieval

            # Response Relevance to Answer
            response_sim = self.compute_similarity(
                self.additional_context_text, self.generative_response
            )
            evaluation["Generation_Similarity_to_Retrieved_Context"] = response_sim

            rouge_scores = self.compute_rouge(
                self.additional_context_text, self.generative_response
            )
            evaluation["Rouge1score_RetrievedContext_vs_Generation"] = rouge_scores["rouge1"].fmeasure
            evaluation["Rouge2score_RetrievedContext_vs_Generation"] = rouge_scores["rouge2"].fmeasure
            evaluation["RougeLscore_RetrievedContext_vs_Generation"] = rouge_scores["rougeL"].fmeasure

            bleu_score = self.compute_bleu(
                self.additional_context_text, self.generative_response
            )
            evaluation["BLEUscore_RetrievedContext_vs_Generation"] = bleu_score

            # TODO: Implement RAGAS

        return evaluation
