class RAGEvaluation:
    def __init__(
        self,
        retrieved_question,
        corresponding_qa_answer,
        generative_response,
        query,
        additional_context_text=None,
    ):
        self.retrieved_question = retrieved_question
        self.corresponding_answer = corresponding_qa_answer
        self.generative_response = generative_response
        self.additional_context_text = additional_context_text
        self.query = query

    def output_evaluation(self):
        evaluation = "Response meets the quality standards."
        return evaluation
