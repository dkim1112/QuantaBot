from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import LLM

class MyOpenAI(LLM):
    def __init__(self, model_name="gpt-4", temperature=0.7): # higher the temp, more creative the output.
        super().__init__()
        self._model = ChatOpenAI(model=model_name, temperature=temperature)

    def _call(self, prompt: str, **kwargs) -> str:
        response = self._model.invoke(prompt)
        return response.content
    
    @property
    def _llm_type(self):
        return "openai"
