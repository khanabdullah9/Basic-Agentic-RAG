from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Sequence, Union, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, InjectedState
import requests
import os
from dotenv import load_dotenv


from vector import collection_exists, create_vector_store, perform_similarity_search
from utils import save_png, log_info

load_dotenv()

api_key = os.environ.get("OLLAMA_API_KEY")

llm = ChatGroq(
    model = "qwen/qwen3.8-27b",
    api_key = os.getenv("OLLAMA_API_KEY")
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    pdf_path: str


@tool
def vector_search(prompt: str, state: Annotated[AgentState, InjectedState]) -> list[str]:
    """Performs similarity search from the vector store

    Args:
        prompt (str): query
        state (Annotated[AgentState, InjectedState]): Agent's state

    Returns:
        list[str]: retrieved context
    """
    return perform_similarity_search(prompt = prompt, pdf_path = state["pdf_path"])
    

tools = [vector_search]
llm_with_tools = llm.bind_tools(tools)

def create_embeddings(state: AgentState) -> AgentState:
    """Creates vector store

    Args:
        state (AgentState): Agent's state

    Returns:
        AgentState: Agent's state
    """
    if collection_exists(state["pdf_path"]):
        log_info("Collection already exists")
        return state
    
    ack = create_vector_store(state["pdf_path"])
    log_info(f"{'Vector DB loaded' if ack else 'Vector DB NOT loaded'}")
    return state

def llm_node(state: AgentState) -> AgentState: # decision node
    sys_msg = SystemMessage(content = """
                You are an helpful assistant!
                You are supposed to generate responses based on retrieved context.
                Invoke the LLM and decide which tool to call.

                List of available tools:
                1. vector_search: Performs similarity search and retrieves context based on the provided prompt.

                Never respond to 'out of context' queries.
                Only generate responses based on the retrieved context.
                Do not re-run the tool if it returns an output.
            """)
    
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    state["messages"].append(response)
    log_info(response)

    return state

def tools_condition(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "tool_node" if last_message.tool_calls else END

graph = StateGraph(AgentState)

graph.add_node("create_embeddings", create_embeddings)
graph.add_node("llm_node", llm_node)
graph.add_node("tool_node", ToolNode(tools = tools))

graph.add_edge(START, "create_embeddings")
graph.add_edge("create_embeddings", "llm_node")
graph.add_conditional_edges("llm_node", tools_condition, {"tool_node": "tool_node", END: END})
graph.add_edge("tool_node","llm_node")

agent = graph.compile()

# save_png(agent)
