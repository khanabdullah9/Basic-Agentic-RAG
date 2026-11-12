from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os
from pprint import pprint

from agent import agent

while True:
    prompt = input("Enter: ")
    if prompt.strip().lower() == "q":
        break
    if prompt.strip().lower() == "cls":
        continue

    response = agent.invoke(dict(
        messages = [HumanMessage(content = prompt)],
        pdf_path = os.path.join("DOCS","PDF","GAN.pdf"),
        excel_path = os.path.join("DOCS","Excel","sales.xlsx"),
    ))
    if response:
        for idx, msg in enumerate(response["messages"]):
            msg.pretty_print()