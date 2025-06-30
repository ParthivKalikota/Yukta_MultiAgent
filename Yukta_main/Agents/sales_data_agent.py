import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
import pandas as pd
import matplotlib.pyplot as plt
import io
from datetime import datetime

_sales_llm = None
DATABASE_URI = None
db_engine = None
sql_agent_executor = None

def init_sales_data_agent(sales_llm, db_uri):
    global _sales_llm, DATABASE_URI, db_engine, sql_agent_executor
    _sales_llm = sales_llm
    DATABASE_URI = db_uri 
    try:
        db_engine = SQLDatabase.from_uri(DATABASE_URI)
        sql_toolkit = SQLDatabaseToolkit(db = db_engine, llm = _sales_llm)
        all_sql_tools = sql_toolkit.get_tools()
        sales_agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                """You are an expert SQL assistant. Your goal is to translate user questions into accurate PostgreSQL queries and execute them using the provided tools.
                You have access to the 'sales' table.
                **Schema for the 'sales' table:**
                {table_info}

                When generating a SQL query, ensure it is correct PostgreSQL syntax and ONLY uses `SELECT` statements.
                DO NOT generate `INSERT`, `UPDATE`, `DELETE`, `DROP`, or any other data-modifying queries.
                For charting requests, generate queries that aggregate data and alias aggregated columns clearly (e.g., SUM(total_sale) AS total_sales).
                Always return the raw data result from the database query.

                You have the following tools available:
                - `sql_db_query(query: str)`: Execute a SQL query against the database.
                - `sql_db_schema(table_names: List[str])`: Get the schema of specified tables.
                - `sql_db_query_checker(query: str)`: Check if a SQL query is syntactically correct and safe to run.

                Always use `sql_db_query_checker` BEFORE `sql_db_query`.
                """
                ),
                MessagesPlaceholder(variable_name="messages"), # <--- IMPORTANT: For conversation history
                MessagesPlaceholder(variable_name="agent_scratchpad"), # <--- IMPORTANT: For ReAct thoughts/actions
            ]
        ).partial(table_info=db_engine.get_table_info(table_names=['sales']))
        sql_agent_executor = AgentExecutor(
            agent=create_tool_calling_agent(_sales_llm, all_sql_tools, sales_agent_prompt),
            tools=all_sql_tools,
            verbose=True,
            handle_parsing_errors=True 
        )
        print("SQL Agent Executor initialized successfully.")
    except Exception as e:
        print(f"Error initializing SQL Agent components: {e}")
        db_engine = None
        sql_agent_executor = None



@tool
def get_data_from_sales(question: str) -> str:
    """
    Generates and executes a SQL query based on the user's question to retrieve data from the 'sales' table.
    Ensures queries are safe and read-only.
    """
    print("\n--- INVOCATION OF GET_DATA_FROM_SALES TOOL ---")
    if sql_agent_executor is None:
        return "SQL data retrieval system not initialized due to a configuration error."

    try:
        # Pass the user's question to the SQL agent executor
        # Use messages format as per ChatPromptTemplate recommendation
        response = sql_agent_executor.invoke({"messages": [HumanMessage(content=question)]})

        # The response structure from AgentExecutor.invoke() varies.
        # It usually returns a dictionary with 'output' or 'messages'.
        # We want the final AI message content.
        if "output" in response and response["output"]:
            return response["output"]
        elif "messages" in response and response["messages"]:
            # Look for the last AI message which should contain the answer
            for msg in reversed(response["messages"]):
                if isinstance(msg, AIMessage) and msg.content.strip():
                    return msg.content
                elif isinstance(msg, ToolMessage) and msg.name == "sql_db_query":
                    # If the last thing was a tool execution, return its content
                    return msg.content
            return "SQL Agent executed but no clear output message found."
        else:
            return "SQL Agent executed but returned an unexpected response format."

    except Exception as e:
        return f"An error occurred during SQL query generation or execution: {e}"
    
@tool
def generate_chart_tool(data_csv: str, chart_type: str, title: str = "Sales Data Chart",
                        x_label: str = None, y_label: str = None,
                        group_by_column: str = None, value_column: str = None) -> str:
    """
    Generates a chart (bar or pie) from data provided as a CSV string.
    This tool is designed to visualize aggregated sales data.

    The 'data_csv' input *must* be a string containing comma-separated values, including a header row.
    If 'group_by_column' and 'value_column' are provided, the data will be grouped by 'group_by_column'
    and the sum of 'value_column' will be used for the chart. These column names MUST exactly match
    the column names in the provided 'data_csv' (e.g., if your SQL returns 'category' and 'sum', use those).

    Args:
        data_csv (str): The data in CSV string format. Example: "category,total_sales\nElectronics,1500\nAccessories,500"
        chart_type (str): The type of chart to generate. Must be 'bar' or 'pie'.
        title (str, optional): The main title of the chart. Defaults to "Sales Data Chart".
        x_label (str, optional): Label for the X-axis (for bar charts).
        y_label (str, optional): Label for the Y-axis (for bar charts).
        group_by_column (str, optional): The column name from 'data_csv' to group by. Required for bar/pie charts.
        value_column (str, optional): The column name from 'data_csv' that contains the numeric values to plot. Required for bar/pie charts.
    Returns:
        str: File path to the generated chart image (e.g., "charts/bar_chart_20250621_143000.png"), or an error message.
    """
    try:
        print(f"\n--- DEBUG: generate_chart_tool received data_csv (first 500 chars) ---\n{data_csv[:500]}...\n-----------------------------------------------\n")

        df = pd.read_csv(io.StringIO(data_csv))
        df.columns = df.columns.str.strip() # Strip whitespace from column names

        if group_by_column and value_column:
            group_by_column = group_by_column.strip()
            value_column = value_column.strip()

            if group_by_column not in df.columns:
                return f"Error: Grouping column '{group_by_column}' not found in data columns: {df.columns.tolist()}."
            if value_column not in df.columns:
                return f"Error: Value column '{value_column}' not found in data columns: {df.columns.tolist()}."

            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
            df.dropna(subset=[value_column], inplace=True)

            if df.empty:
                return "Error: No valid numeric data found for charting after processing."

            chart_data = df.groupby(group_by_column)[value_column].sum().sort_values(ascending=False)
        else:
            return "Error: Both 'group_by_column' and 'value_column' must be provided for bar/pie charts. Ensure the LLM provides these."

        plt.figure(figsize=(10, 6))

        if chart_type == 'bar':
            chart_data.plot(kind='bar', color='skyblue')
            plt.title(title)
            if x_label: plt.xlabel(x_label)
            if y_label: plt.ylabel(y_label)
            plt.xticks(rotation=45, ha='right')
        elif chart_type == 'pie':
            if chart_data.sum() == 0:
                return "Error: Cannot generate pie chart. All values are zero or missing after aggregation."
            chart_data.plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap='Pastel1')
            plt.title(title)
            plt.ylabel('')
        else:
            return "Error: Unsupported chart type. Choose 'bar' or 'pie'."

        plt.tight_layout()
        charts_dir = "charts"
        os.makedirs(charts_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_filename = f"{charts_dir}/{chart_type}_chart_{timestamp}.png"
        plt.savefig(chart_filename)
        plt.close()

        return f"Chart generated successfully: {chart_filename}"

    except Exception as e:
        return f"An error occurred while generating the chart: {e}"
    
sales_data_agent_prompt = """You are a specialized Sales Data Analyst Agent.
        Your goal is to answer questions about sales data and generate visualizations when explicitly requested.
        You have access to a PostgreSQL sales database and charting capabilities.

        **Here are your available tools:**
        1.  `get_data_from_sales_tool(question: str)`: Use this tool to query the PostgreSQL database for sales data.
            The input to this tool is the user's specific question about sales. It returns raw data (e.g., CSV string).
            *Important*: If the user asks for a chart, ensure your input to this tool (the 'question') results in aggregated data suitable for charting (e.g., "get total sales by category", "sum of sales per region").
        2.  `generate_chart_tool(data_csv: str, chart_type: str, title: str, x_label: str, y_label: str, group_by_column: str, value_column: str)`: Use this tool to create a bar or pie chart.
            It requires `data_csv` (from `get_data_from_sales_tool`), `chart_type`, `title`, `x_label`, `y_label`, `group_by_column`, and `value_column`.
            You MUST infer `chart_type`, `title`, `x_label`, `y_label`, `group_by_column`, and `value_column` from the original user's request AND the column names present in the `data_csv` (which you will observe from `get_data_from_sales_tool`'s output).

        **Workflow Instructions:**
        -   **If the user asks for a chart (e.g., "bar chart", "pie chart", "visualize", "plot", "graph"):**
            -   **Step A: Get Raw Data.** First, use the `get_data_from_sales_tool`. Formulate the `question` for this tool to retrieve aggregated data relevant to the charting request.
            -   **Step B: Infer Chart Parameters.** Once you receive the raw data result (from `get_data_from_sales_tool`), carefully analyze the original user's question AND the column headers/structure of the received data. Infer the `chart_type` (must be 'bar' or 'pie'), an appropriate `title`, `x_label`, `y_label`, and crucially, the exact `group_by_column` and `value_column` names *from the data's headers*.
            -   **Step C: Generate Chart.** Then, call `generate_chart_tool` with the data (as `data_csv`) and all the parameters you inferred.
            -   **Step D: Final Answer.** The output of `generate_chart_tool` will be the chart's file path. Present this file path as your final answer to the supervisor. Do NOT add any extra conversational text.

        -   **If the user asks for data retrieval or analysis that does NOT require a chart:**
            -   Use `get_data_from_sales_tool` directly with the user's question.
            -   Present the raw data or a concise summary of it as your final answer to the user.

        -   Always ensure your final answer is clear and directly addresses the user's request.
        """

def create_sales_data_agent():
    sales_data_agent = create_react_agent(
        model = _sales_llm,
        tools = [get_data_from_sales, generate_chart_tool],
        prompt = sales_data_agent_prompt,
        name = "SalesDataAgent"
    )
    return sales_data_agent

