import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1. Load environment variables (.env contains Groq and Langfuse API keys)
load_dotenv()

# 2. Initialize Langfuse Callback Handler
# It automatically picks up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL from the environment
langfuse_handler = CallbackHandler()

# 3. Initialize Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# 4. Define State for the Multi-Agent workflow
class AgentState(TypedDict):
    topic: str
    ideas: str
    story: str

# 5. Define Agents (Nodes)

# Agent 1: Ideator
def ideator_node(state: AgentState):
    print("-> Ideator Agent is brainstorming ideas...")
    topic = state["topic"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative brainstorming assistant. Generate 3 unique plot ideas for a story about the given topic. Keep them concise."),
        ("human", "{topic}")
    ])
    chain = prompt | llm
    # The callback handler is injected automatically at the graph level, 
    # but we could also pass config={"callbacks": [langfuse_handler]} here manually.
    response = chain.invoke({"topic": topic})
    return {"ideas": response.content}

# Agent 2: Writer
def writer_node(state: AgentState):
    print("-> Writer Agent is drafting the story...")
    ideas = state["ideas"]
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert storyteller. Write a short, engaging story (max 200 words) based on the provided ideas."),
        ("human", "Ideas:\n{ideas}")
    ])
    chain = prompt | llm
    response = chain.invoke({"ideas": ideas})
    return {"story": response.content}

# 6. Build the Multi-Agent Graph
workflow = StateGraph(AgentState)

workflow.add_node("ideator", ideator_node)
workflow.add_node("writer", writer_node)

workflow.add_edge(START, "ideator")
workflow.add_edge("ideator", "writer")
workflow.add_edge("writer", END)

# Compile the workflow
app = workflow.compile()

if __name__ == "__main__":
    print("=== Starting Multi-Agent Workflow ===\n")
    initial_state = {"topic": "A detective who solves crimes by communicating with house plants."}
    
    # 7. Run the graph and pass the langfuse handler to trace the overall execution
    result = app.invoke(initial_state, config={"callbacks": [langfuse_handler]})
    
    print("\n=== GENERATED STORY ===")
    print(result.get("story"))
    
    # 8. Traces are automatically sent.
    print("\n=== Execution Complete: Traces successfully sent to Langfuse! ===")
