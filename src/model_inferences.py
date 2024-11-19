from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingClass:
    def __init__(self, model_name):
        self.model_name = model_name
        model_kwargs = {"trust_remote_code": "True"}
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs
        )


class InstructClass:
    def __init__(self, config):
        self.model_path = config["model"]["llm_model_path"]
        self.n_ctx = config["model"]["ctx_len"]
        self.n_threads = config["model"]["no_of_threads"]
        self.temperature = config["model"]["temperature"]
        self.max_tokens = config["model"]["max_tokens"]
        self.repeat_penalty = config["model"]["repeat_penalty"]
        self.top_p = config["model"]["top_p"]

        self.instruct_llm = ChatLlamaCpp(
            temperature=self.temperature,
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            max_tokens=self.max_tokens,
            n_threads=self.n_threads,
            repeat_penalty=self.repeat_penalty,
            top_p=self.top_p,
            verbose=True,
        )

    def judge_chain(self):
        judge_prompt = PromptTemplate.from_template(
            """
            Given the user's query: "{query}"
            And the retrieved question: "{retrieved_question}"

            Determine if the retrieved question is relevant to the query.
            Respond with "Relevant" if the retrieved question is relevant, or respond with "False" if the retrieved question is irrelevant.
            Be strict.
            
            <<<
            Response:
            >>>
            """
        )

        self.judge_inf = judge_prompt | self.instruct_llm

    def response_chain(self):
        response_prompt = PromptTemplate.from_template(
            """
            # Based on the following context only:
            
            {context}
            
            # Provide a short and concise answer to the user's query: "{query}"

            <<<
            Answer:
            >>>
            """
        )
        self.response_inf = response_prompt | self.instruct_llm
