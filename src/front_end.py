import asyncio
import logging

import gradio as gr

from evaluation import RAGEvaluation

logger = logging.getLogger(__name__)

# Async version of the front-end.


class GradioInterface:
    def __init__(self, instruct_class, rag_system, config):
        self.rag_system = rag_system
        self.title = config["gradio"]["title"]
        self.instruct_llm = instruct_class.instruct_llm

    async def gradio_RAGLogic(self, query, retrieval_only=False, evaluate=True):
        try:
            retrieved_question, corresponding_qa_answer = await asyncio.to_thread(
                self.rag_system.retrieval_qa, query
            )
            yield corresponding_qa_answer, "Processing...", "Processing...", "Processing..."
        except Exception as e:
            logger.error(f"Error in retrieving answer: {str(e)}")
            yield "Error in retrieval. Process has Terminated.", "", "", ""
            return

        try:
            judge_output = await asyncio.to_thread(
                self.rag_system.generation_judge, retrieved_question, query
            )
            logger.info(f"Judge Output: {judge_output}")
        except Exception as e:
            logger.error(f"Error in Judging Context: {str(e)}")
            relevancy_statement = "Error in Judging Output. Process has Terminated."
            yield corresponding_qa_answer, relevancy_statement, "", ""
            return

        if "Relevant" in judge_output:
            relevancy_statement = "Retrieved Context Is Relevant to The Query"
            yield corresponding_qa_answer, relevancy_statement, "Processing...", "Processing..."
            if not retrieval_only:
                generation_context_qa = (
                    retrieved_question + " " + corresponding_qa_answer
                )
                try:
                    generative_response = await asyncio.to_thread(
                        self.rag_system.generation_response,
                        generation_context_qa,
                        query,
                    )
                    yield corresponding_qa_answer, relevancy_statement, generative_response, "Processing..."
                except Exception as e:
                    logger.error(f"Error in Generating Output: {str(e)}")
                    generative_response = (
                        "Error in Generating Output. Process has Terminated."
                    )
                    yield corresponding_qa_answer, relevancy_statement, generative_response, ""
                    return

                if evaluate:
                    evaluation = await asyncio.to_thread(
                        RAGEvaluation(
                            retrieved_question,
                            corresponding_qa_answer,
                            generative_response,
                            query,
                            self.instruct_llm,
                        ).output_evaluation
                    )
                    yield corresponding_qa_answer, relevancy_statement, generative_response, evaluation
                else:
                    evaluation = "No evaluation carried out. Please set evaluation toggle to True if required."
                    yield corresponding_qa_answer, relevancy_statement, generative_response, evaluation
            else:
                evaluation = (
                    "No evaluation carried out as 'Retrieval Only' was set to True."
                )
                generative_response = "Answer was not generated using our LLM. Please set Retrieval Only toggle to False if required."
                yield corresponding_qa_answer, relevancy_statement, generative_response, evaluation
        else:
            relevancy_statement = "QA Pair Retrieval Was Not Relevant to The Query. Searched in Extended Context."
            try:
                context_text = await asyncio.to_thread(
                    self.rag_system.retrieval_added_context, query
                )
                yield context_text, relevancy_statement, "Processing...", "Processing..."
            except Exception as e:
                logger.error(f"Error in retrieving additional context: {str(e)}")
                yield "Error in additional context retrieval. Process has Terminated.", "", "", ""
                return

            if not retrieval_only:
                try:
                    generative_response = await asyncio.to_thread(
                        self.rag_system.generation_response, context_text, query
                    )
                    yield context_text, relevancy_statement, generative_response, "Processing..."
                except Exception as e:
                    logger.error(f"Error in Generating Output: {str(e)}")
                    generative_response = (
                        "Error in Generating Output. Process has Terminated."
                    )
                    yield corresponding_qa_answer, relevancy_statement, generative_response, ""
                    return
                if evaluate:
                    evaluation = await asyncio.to_thread(
                        RAGEvaluation(
                            retrieved_question,
                            corresponding_qa_answer,
                            generative_response,
                            query,
                            self.instruct_llm,
                            context_text,
                        ).output_evaluation
                    )
                    yield context_text, relevancy_statement, generative_response, evaluation
                else:
                    evaluation = "No evaluation carried out. Please set evaluation toggle to True if required."
                    yield context_text, relevancy_statement, generative_response, evaluation
            else:
                generative_response = "Answer was not generated using our LLM. Please set Retrieval Only toggle to False if required."
                evaluation = (
                    "No evaluation carried out as 'Retrieval Only' was set to True."
                )
                yield context_text, relevancy_statement, generative_response, evaluation

    def launch(self):
        iface = gr.Interface(
            fn=self.gradio_RAGLogic,
            inputs=[
                gr.Textbox(label="User Query", placeholder="Enter your question!"),
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

        iface.launch(debug=True)
