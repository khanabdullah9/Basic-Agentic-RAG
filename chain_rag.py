from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

from vector import load_vector_store
from utils import log_info

load_dotenv()

llm = ChatGroq(
    model = "qwen/qwen3.8-27b",
    api_key = os.getenv("OLLAMA_API_KEY")
)
log_info("llm loaded")

example_prompt = ChatPromptTemplate.from_messages([
    ("user","{prompt}"),
    ("ai","{output}")
])
examples = [
    {
        "prompt": "What is the full form of LLM?",
        "output": "Based on the retrieved context: LLM stands for Large Language Model."
    }
]

few_shot = FewShotChatMessagePromptTemplate(
    example_prompt = example_prompt,
    examples = examples
)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are an helpful assistant!"),
    ("system","Based on the retrieved context respond to the user queries."),
    ("system","Retrieved context: {context}"),
    few_shot,
    ("user","{user_prompt}"),
])

vector_store = load_vector_store(os.path.join("DOCS","PDF","GAN.pdf"))
retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":3})
log_info("retriever loaded")

def format_docs(documents) -> str:
    return "\n\n".join([docs.page_content for docs in documents])

chain = (
    {
        "context": retriever | format_docs,
        "user_prompt": RunnablePassthrough() 
    } 
    | prompt 
    | llm
    )

response = chain.invoke("What is this document about?")
response.pretty_print()

