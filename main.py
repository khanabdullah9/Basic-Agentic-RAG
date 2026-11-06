from langchain_core.messages import HumanMessage
import os

from agent import agent

response = agent.invoke(dict(
    messages = [HumanMessage(content = "What is this doc about?")],
    pdf_path = os.path.join("DOCS","HP1.pdf")
))
print(response)