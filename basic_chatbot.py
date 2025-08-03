from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your Google API key here if not already set in your environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

CONFIG={"configurable":{"thread_id":"thread_1"}}

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GOOGLE_API_KEY
)

class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chatbot(state: ChatbotState):
    messages = state["messages"]
    if not messages:
        messages.append(HumanMessage(content="Hello!"))
    response = llm.invoke(messages,config=CONFIG)
    messages.append(response)
    return {"messages": messages}

# Create the graph
checkpointer = MemorySaver()
graph = StateGraph(ChatbotState)

# add nodes
graph.add_node('chat_node', chatbot)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer) 