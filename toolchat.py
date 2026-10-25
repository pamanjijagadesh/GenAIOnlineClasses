import json
import os
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# --------------------------------------------------------------------------- #
# 1. Load the data
# --------------------------------------------------------------------------- #
CSV_PATH = "StudentsDataSet.csv"
SHOW_TOOL_CALLS = True

df = pd.read_csv(CSV_PATH)
# --------------------------------------------------------------------------- #
# 2. System Prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions about a student dataset.

Dataset columns: {list(df.columns)}
Rows: {len(df)}

Rules:
- ALWAYS use the tools to get data. Never guess or invent numbers.
- Do the reasoning yourself: combine tool results, compute ratios/percentages, and explain clearly.
- The dataset only contains attendance information and whether a student is debarred.
  It has NO marks, subjects, or pass/fail columns. If the user asks about those,
  say clearly that the data does not contain them, and offer the closest thing you
  can answer (e.g. debarred vs. not debarred, attendance).
- Do not treat "debarred" as "failed" unless the user says so; mention it as an assumption if you offer it.
- Keep answers short and direct.
"""
# --------------------------------------------------------------------------- #
# 3. Tools  (they only fetch data - the LLM does the thinking)
# --------------------------------------------------------------------------- #
@tool
def dataset_info() -> str:
    """Get an overview of the dataset: row count, column names and a few sample rows.
    Call this if you are unsure what data is available."""
    return json.dumps({
        "rows": len(df),
        "columns": list(df.columns),
        "gender_values": df["Gender"].unique().tolist(),
        "is_debarred_values": df["Is Debarred"].unique().tolist(),
        "sample_rows": df.head(3).to_dict("records"),
    })


@tool
def find_student(query: str) -> str:
    """Look up an individual student by UID (e.g. '23msm40003') or by name
    (full or partial, case-insensitive, e.g. 'priya'). Returns the full record(s)."""
    q = query.strip().lower()
    matches = df[
        df["UID"].str.lower().str.contains(q) | df["Name"].str.lower().str.contains(q)
    ]
    if matches.empty:
        return f"No student found for '{query}'."
    return json.dumps(matches.to_dict("records"))


@tool
def filter_students(
    gender: Optional[str] = None,
    is_debarred: Optional[str] = None,
    min_attendance: Optional[int] = None,
    max_attendance: Optional[int] = None,
) -> str:
    """List students that match ALL the conditions given. Leave a condition empty to ignore it.
    Returns the number of matching students and their records.

    - gender: 'Male' or 'Female'
    - is_debarred: 'Yes' or 'No'
    - min_attendance / max_attendance: attendance percentage range (inclusive), e.g. 75
    """
    result = df
    if gender:
        result = result[result["Gender"].str.lower() == gender.lower()]
    if is_debarred:
        result = result[result["Is Debarred"].str.lower() == is_debarred.lower()]
    if min_attendance is not None:
        result = result[result["Attendence Percentage"] >= min_attendance]
    if max_attendance is not None:
        result = result[result["Attendence Percentage"] <= max_attendance]

    return json.dumps({
        "total_matches": len(result),
        "students": result.to_dict("records"),
    })


@tool
def summary_by_group(group_by: Literal["Gender", "Is Debarred"]) -> str:
    """Get the number of students and the attendance statistics (average, min, max)
    for each group. Use group_by='Gender' to compare male vs female, or
    group_by='Is Debarred' to compare debarred vs not debarred students."""
    summary = df.groupby(group_by)["Attendence Percentage"].agg(["count", "mean", "min", "max"])
    return summary.round(2).to_json(orient="index")

TOOLS = [dataset_info, find_student, filter_students, summary_by_group]

# --------------------------------------------------------------------------- #
# 4. LangGraph:  START -> agent -> (tools -> agent)* -> END
# --------------------------------------------------------------------------- #
def build_graph():
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    # Node 1: the LLM decides - answer directly, or call a tool
    def agent(state: MessagesState):
        reply = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [reply]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))  # Node 2: runs the tool the LLM asked for

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tool call -> "tools", else -> END
    graph.add_edge("tools", "agent")  # after a tool runs, go back to the LLM

    return graph.compile(checkpointer=MemorySaver())  # memory, so follow-up questions work


# --------------------------------------------------------------------------- #
# 5. Chat loop
# --------------------------------------------------------------------------- #
def main():
    app = build_graph()
    config = {"configurable": {"thread_id": "student-chat"}}

    print(f"Student Data Chatbot ({len(df)} students loaded). Type 'exit' to quit.\n")
    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit", "q"}:
            break

        final = ""
        for step in app.stream({"messages": [HumanMessage(user)]}, config, stream_mode="values"):
            msg = step["messages"][-1]
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    if SHOW_TOOL_CALLS:
                        for call in msg.tool_calls:
                            print(f"  [tool] {call['name']}({json.dumps(call['args'])})")
                else:
                    final = msg.content
        print(f"\nBot: {final}\n")

    print("Bye!")


if __name__ == "__main__":
    main()