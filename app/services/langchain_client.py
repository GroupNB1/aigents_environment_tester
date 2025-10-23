from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

class LangchainClient:
    def __init__(self, model_name: str):
        self.llm = OpenAI(model_name=model_name)

    def generate_response(self, prompt: str) -> str:
        template = PromptTemplate(input_variables=["input"], template=prompt)
        chain = LLMChain(llm=self.llm, prompt=template)
        response = chain.run(input=prompt)
        return response