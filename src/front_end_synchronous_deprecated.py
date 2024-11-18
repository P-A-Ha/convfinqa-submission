import logging

import gradio as gr

from evaluation import RAGEvaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradioInterface:
    def __init__(self, rag_system, config):
        self.rag_system = rag_system
        self.title = config["gradio"]["title"]

    def gradio_RAGLogic(self, query, retrieval_only, evaluate=True):
        retrieved_question, corresponding_qa_answer = self.rag_system.retrieval_qa(
            query
        )
        judge_output = self.rag_system.generation_judge(retrieved_question, query)
        generative_response = None
        if "Relevant" in judge_output:
            logger.info("Retrieved Question is relevant.")
            if ~retrieval_only:
                generative_response = self.rag_system.generation_response(
                    corresponding_qa_answer, query
                )
                if evaluate:
                    evaluation = RAGEvaluation(
                        retrieved_question,
                        corresponding_qa_answer,
                        generative_response,
                        query,
                    ).output_evaluation()
                else:
                    evaluation = "No evaluation carried out. Please set evaluation toggle to True if required."
            else:
                generative_response = "Answer was not generated usin our LLM. Please set Retrieval Only toggle to False if required."

            relevancy_statement = "Retrieved Context Is Relevant to The Query"
            return (
                corresponding_qa_answer,
                relevancy_statement,
                generative_response,
                evaluation,
            )
        else:
            context_text = self.rag_system.retrieval_added_context(query)
            if ~retrieval_only:
                generative_response = self.rag_system.generation_response(
                    context_text, query
                )
                if evaluate:
                    evaluation = RAGEvaluation(
                        retrieved_question,
                        corresponding_qa_answer,
                        generative_response,
                        query,
                        context_text,
                    ).output_evaluation()
                else:
                    evaluation = "No evaluation carried out. Please set evaluation toggle to True if required."
            else:
                generative_response = "Answer was not generated using our LLM. Please set Retrieval Only toggle to False if required."
            relevancy_statement = "Retrieved Context Is Not Relevant to The Query"
            return context_text, relevancy_statement, generative_response, evaluation

    def launch(self):
        iface = gr.Interface(
            fn=self.gradio_RAGLogic,
            inputs=[
                gr.Textbox(label="Question", placeholder="Enter your query"),
                gr.Checkbox(label="Retrieval Only"),
                gr.Checkbox(label="Evaluate Output"),
            ],
            outputs=[
                gr.Textbox(label="Retrieval Output"),
                gr.Textbox(label="Relevancy To Query"),
                gr.Textbox(label="Generative Output"),
                gr.Textbox(label="Evaluation Output"),
            ],
            title=self.title,
            description="FinQA RAG Application",
        )
        iface.launch()
