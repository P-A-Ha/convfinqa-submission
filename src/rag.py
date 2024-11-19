import logging

from langchain_community.chat_models import ChatLlamaCpp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGComponents:
    def __init__(
        self,
        instruct_class: ChatLlamaCpp,
        vector_store_mngr,
        batch_eval_retrieval=False,
    ):
        self.vector_store_mngr = vector_store_mngr
        self.qa_retriever = self.vector_store_mngr.get_qa_retriever()
        self.context_retriever = self.vector_store_mngr.get_meta_retriever()
        if not batch_eval_retrieval:
            self.instruct_class = instruct_class
            self.instruct_class.judge_chain()
            self.instruct_class.response_chain()

    def retrieval_qa(self, query):
        logger.info("Retrieving Documents")
        retrieved_docs = self.qa_retriever.get_relevant_documents(query)
        retrieved_question = retrieved_docs[0].page_content
        corresponding_answer = retrieved_docs[0].metadata["answer"]
        logger.info(f"Retrieved Question: {retrieved_question}")
        logger.info(f"Retrieved Answer: {corresponding_answer}")
        return retrieved_question, corresponding_answer

    def retrieval_added_context(self, query):
        logger.info("Retrieving Documents")
        retrieved_docs = self.context_retriever.get_relevant_documents(query)
        context_text = " ".join([doc.page_content for doc in retrieved_docs])
        return context_text

    def generation_judge(self, retrieved_question, query):
        logger.info("Judging Relevance of Query to Retrieved Question")
        judge_output = self.instruct_class.judge_inf.invoke(
            {"query": query, "retrieved_question": retrieved_question}
        )
        logger.info(f"Judge Output: {judge_output.content}")
        return judge_output.content

    def generation_response(self, additional_context_text, query):
        logger.info("Generating response.")
        response = self.instruct_class.response_inf.invoke(
            {"query": query, "context": additional_context_text}
        )
        logger.info(f"Response Output: {response.content}")

        return response.content
