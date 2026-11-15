"""
Tool-calling RAG chatbot with chat history (LangChain + LangGraph + Groq).

Install:
    pip install langgraph langchain-groq langchain-core python-dotenv

Put your key in a .env file:
    GROQ_API_KEY=your_key_here

Files needed in the same folder:
    rag1.py  -> Machine Learning PDF
    rag2.py  -> Data Visualization PDF
    rag3.py  -> Python PDF
Each file must have the function: get_relevant_info(question)

How it works:
    You ask a question
        -> the LLM decides which tool (PDF) to search
        -> the tool returns text from the PDF
        -> the LLM answers using ONLY that text
    LangGraph's MemorySaver remembers the conversation (chat history).
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import the search function from each RAG file
from rag1 import get_relevant_info as search_ml_pdf
from rag2 import get_relevant_info as search_dataviz_pdf
from rag3 import get_relevant_info as search_python_pdf


load_dotenv()


# ---------- Step 1: The LLM ----------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0,
)


# ---------- Step 2: The tools ----------
# The docstring (text inside """ """) tells the LLM when to use the tool.
# The print() line shows us in the terminal that the tool was called.

@tool
def machine_learning_tool(question: str) -> str:
    """Search the Machine Learning PDF.
    Use this for questions about machine learning: algorithms, regression,
    classification, clustering, model training, overfitting, neural networks."""
    print(f"\n[TOOL CALLED] machine_learning_tool | question: {question}")
    results = search_ml_pdf(question)
    return "\n\n".join(results)


@tool
def data_visualization_tool(question: str) -> str:
    """Search the Data Visualization PDF.
    Use this for questions about data visualization: charts, plots, graphs,
    colours, dashboards, choosing the right chart type."""
    print(f"\n[TOOL CALLED] data_visualization_tool | question: {question}")
    results = search_dataviz_pdf(question)
    return "\n\n".join(results)


@tool
def python_tool(question: str) -> str:
    """Search the Python PDF.
    Use this for questions about the Python language: syntax, variables,
    loops, functions, classes, modules, file handling, error handling."""
    print(f"\n[TOOL CALLED] python_tool | question: {question}")
    results = search_python_pdf(question)
    return "\n\n".join(results)


tools = [machine_learning_tool, data_visualization_tool, python_tool]

# Tell the LLM that these tools exist
llm_with_tools = llm.bind_tools(tools)


# ---------- Step 3: The system prompt ----------
SYSTEM_PROMPT = """You are a document assistant. You answer ONLY from the
information returned by your tools. You must NEVER use your own knowledge.

Your tools (each one searches one PDF):
1. machine_learning_tool    -> Machine Learning PDF
2. data_visualization_tool  -> Data Visualization PDF
3. python_tool              -> Python PDF

Rules:
1. For every question, call the correct tool first. If the question covers
   more than one subject, call more than one tool.
2. Write your answer using ONLY the text the tool returned.
3. If the tool text does not contain the answer, reply exactly:
   "I could not find this in the documents."
4. If the question is not about Machine Learning, Data Visualization or
   Python, reply exactly:
   "I can only answer questions from the Machine Learning, Data Visualization
   and Python documents."
5. For a simple greeting like "hi" or "thanks", reply with a short greeting and
   say what you can help with. Do not use a tool.
6. Use the earlier messages in the chat to understand follow-up questions
   like "explain that more". For a follow-up, still call a tool again."""


# ---------- Step 4: The graph node ----------
def assistant(state: MessagesState):
    """The LLM node: reads the chat so far and decides what to do next."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ---------- Step 5: Build the graph ----------
#
#   START -> assistant --(needs a tool?)--> tools -> assistant
#                      \--(no)-----------> END
#
builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)  # goes to "tools" or END
builder.add_edge("tools", "assistant")                       # after a tool, LLM answers

# MemorySaver stores the chat history
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# ---------- Step 6: Chat loop ----------
if __name__ == "__main__":
    # thread_id = one conversation. A new id starts a fresh chat.
    config = {"configurable": {"thread_id": "chat-1"}}

    print("\nStudy chatbot is ready! Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Bye!")
            break

        if user_input == "":
            continue

        # Send only the NEW message; LangGraph adds the old history automatically
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # The last message is the final answer from the LLM
        print("\nBot:", result["messages"][-1].content, "\n")