"""
Student Data Chatbot  (LangChain + LangGraph)

- The LLM does all the reasoning.
- Tools only fetch data from the CSV and hand it back to the LLM.
- Run:  python chatbot.py
"""

import json
import operator
import os
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

load_dotenv()

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CSV_PATH = os.getenv("CSV_PATH", str(Path(__file__).parent / "StudentsDataSet.csv"))
SHOW_TOOL_CALLS = os.getenv("SHOW_TOOL_CALLS", "1") == "1"

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

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
- If a tool returns an error, read it and retry with corrected arguments.
- Keep answers short and direct.
"""

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
OPS = {
    "==": operator.eq, "!=": operator.ne,
    ">": operator.gt, ">=": operator.ge,
    "<": operator.lt, "<=": operator.le,
}


class Filter(BaseModel):
    column: str = Field(description="Exact column name from the dataset")
    operator: Literal["==", "!=", ">", ">=", "<", "<=", "contains"]
    value: Union[str, int, float]


def _col(name: str) -> str:
    """Match a column name case-insensitively."""
    for c in df.columns:
        if c.lower() == name.strip().lower():
            return c
    raise ValueError(f"Unknown column '{name}'. Valid columns: {list(df.columns)}")


def _apply_filters(filters: list[Filter]) -> pd.DataFrame:
    out = df
    for f in filters:
        col = _col(f.column)
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            if f.operator == "contains":
                raise ValueError(f"'contains' only works on text columns, '{col}' is numeric.")
            out = out[OPS[f.operator](s, float(f.value))]
        else:
            s = s.astype(str).str.strip().str.lower()
            v = str(f.value).strip().lower()
            if f.operator == "contains":
                out = out[s.str.contains(v, regex=False)]
            elif f.operator in ("==", "!="):
                out = out[OPS[f.operator](s, v)]
            else:
                raise ValueError(f"Operator '{f.operator}' is not valid for text column '{col}'.")
    return out


def _json(obj) -> str:
    return json.dumps(obj, default=str)


# --------------------------------------------------------------------------- #
# Tools  (they only fetch data - the LLM does the thinking)
# --------------------------------------------------------------------------- #
@tool
def dataset_info() -> str:
    """Get the dataset overview: columns, data types, row count, allowed values of
    text columns (e.g. Gender, Is Debarred) and min/max of numeric columns.
    Call this if you are unsure what data is available."""
    info = {"rows": len(df), "columns": {}}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            info["columns"][c] = {"type": "number", "min": df[c].min(), "max": df[c].max()}
        elif df[c].nunique() <= 10:
            info["columns"][c] = {"type": "text", "values": df[c].unique().tolist()}
        else:
            info["columns"][c] = {"type": "text", "example": df[c].iloc[0]}
    return _json(info)


@tool
def find_student(query: str) -> str:
    """Look up individual student(s) by UID (e.g. '23msm40001') or by name
    (full or partial, case-insensitive, e.g. 'priya'). Returns the full record(s)."""
    q = query.strip().lower()
    by_uid = df[df["UID"].str.lower() == q]
    result = by_uid if not by_uid.empty else df[
        df["UID"].str.lower().str.contains(q, regex=False)
        | df["Name"].str.lower().str.contains(q, regex=False)
    ]
    if result.empty:
        return _json({"matches": 0, "message": f"No student found for '{query}'."})
    return _json({"matches": len(result), "students": result.head(10).to_dict("records")})


@tool
def filter_students(filters: list[Filter], limit: int = 20) -> str:
    """List students matching ALL given filters (AND). Returns the total match count
    plus up to `limit` full records.
    Example: male and debarred -> filters=[{"column":"Gender","operator":"==","value":"Male"},
    {"column":"Is Debarred","operator":"==","value":"Yes"}]"""
    try:
        res = _apply_filters(filters)
    except ValueError as e:
        return _json({"error": str(e)})
    return _json({
        "total_matches": len(res),
        "returned": min(len(res), limit),
        "students": res.head(limit).to_dict("records"),
    })


@tool
def aggregate_students(
    operation: Literal["count", "mean", "min", "max", "sum"],
    column: Optional[str] = None,
    group_by: Optional[str] = None,
    filters: Optional[list[Filter]] = None,
) -> str:
    """Compute a statistic over the (optionally filtered) students.
    - operation 'count' needs no column; the others need a numeric `column`.
    - group_by splits the result per value of a column (e.g. 'Gender' or 'Is Debarred').
    Examples: gender counts -> operation='count', group_by='Gender'.
    Average attendance of debarred students -> operation='mean', column='Attendence Percentage',
    filters=[{"column":"Is Debarred","operator":"==","value":"Yes"}]."""
    try:
        data = _apply_filters(filters or [])
        gb = _col(group_by) if group_by else None
        if operation == "count":
            res = data.groupby(gb).size().to_dict() if gb else len(data)
        else:
            if not column:
                return _json({"error": f"'{operation}' needs a numeric column."})
            c = _col(column)
            if not pd.api.types.is_numeric_dtype(data[c]):
                return _json({"error": f"'{c}' is not numeric."})
            grouped = data.groupby(gb)[c] if gb else data[c]
            res = getattr(grouped, operation)()
            res = res.round(2).to_dict() if gb else round(float(res), 2)
    except ValueError as e:
        return _json({"error": str(e)})
    return _json({"operation": operation, "column": column, "group_by": group_by,
                  "rows_considered": len(data), "result": res})


TOOLS = [dataset_info, find_student, filter_students, aggregate_students]


# --------------------------------------------------------------------------- #
# LangGraph:  START -> agent -> (tools -> agent)* -> END
# --------------------------------------------------------------------------- #
def build_graph(llm=None):
    llm = llm or ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-120b",
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent(state: MessagesState):
        reply = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [reply]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)  # tool call -> "tools", else END
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=MemorySaver())  # memory so follow-up questions work


def _text(content) -> str:
    """Some providers return a list of content blocks instead of a string."""
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content


# --------------------------------------------------------------------------- #
# Chat loop
# --------------------------------------------------------------------------- #
def main():
    app = build_graph()
    config = {"configurable": {"thread_id": "student-chat"}}

    print(f"Student Data Chatbot ({len(df)} students loaded). Type 'exit' to quit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit", "q"}:
            break

        final = ""
        for step in app.stream({"messages": [HumanMessage(user)]}, config, stream_mode="values"):
            msg = step["messages"][-1]
            if isinstance(msg, AIMessage):
                if msg.tool_calls and SHOW_TOOL_CALLS:
                    for tc in msg.tool_calls:
                        print(f"  [tool] {tc['name']}({json.dumps(tc['args'])})")
                elif not msg.tool_calls:
                    final = _text(msg.content)
        print(f"\nBot: {final}\n")

    print("Bye!")


if __name__ == "__main__":
    main()

