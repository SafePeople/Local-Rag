from langchain_community.llms.ollama import Ollama
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tools
from langchain.agents import AgentType
from transformers import pipeline

# Initialize llama3.1 model
llama_llm = Ollama(model="llama3.1")

# Create a prompt template
prompt = PromptTemplate(
    input_variables = ["query"],
    template = "Here is the answer to your question: {query}"
)

# Create an LLM chain
llm_chain = LLMChain(llm=llama_llm, prompt=prompt)

# Initialize the sentiment analysis model
sentiment_analysis = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    revision="af0f99b"
)

class IntelligentAgentWithSentiment:
    def __init__(self, llm_chain):
        self.llm_chain = llm_chain
        self.memory = {}
        self.last_question = None
        # Initialize sentiment analysis
        self.sentiment_analysis = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            revision="af0f99b"
        )