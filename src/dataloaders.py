import json
import logging

import yaml
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigLoad:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)


class QAdataloader:
    def __init__(self, config):
        self.qa_file_path = config["data"]["qa_file_path"]
        self.additional_context_path = config["data"]["additional_context_path"]
        self.chunk_size = config["data"]["chunk_size"]
        self.chunk_overlap = config["data"]["chunk_overlap"]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def load_qa(self):
        with open(self.qa_file_path, "r") as file:
            qa_pairs_json = json.load(file)
            qa_pairs = [
                Document(
                    page_content=pair["question"], metadata={"answer": pair["answer"]}
                )
                for pair in qa_pairs_json
            ]
        return qa_pairs

    def load_additional_context(self):
        with open(self.additional_context_path, "r") as file:
            context_docs_json = json.load(file)

        max_length = max(len(doc) for doc in context_docs_json)
        context_docs = [Document(page_content=doc) for doc in context_docs_json]

        if max_length > self.chunk_size:
            logger.info(f"Max Length Exceeds Chunk Size: {max_length}")
            try:
                additional_context_texts = self.text_splitter.split_documents(
                    context_docs
                )
            except ValueError as V:
                logger.error(f"Error in splitting context documents: {str(V)}")

            return additional_context_texts
        else:
            return context_docs
