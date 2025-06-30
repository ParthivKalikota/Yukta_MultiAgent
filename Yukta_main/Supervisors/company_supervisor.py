from langgraph_supervisor import create_supervisor

_llm = None
_sales_data_agent_instance = None


def init_company_supervisor(llm_model, sales_data_agent_obj):
    """
    Initializes global dependencies for the Company Supervisor.
    This function should be called once from yukta_nexus.py.
    """
    global _llm, _sales_data_agent_instance
    _llm = llm_model
    _sales_data_agent_instance = sales_data_agent_obj

company_supervisor_prompt = """
You are the Company Supervisor within the 'Yukta' AI Assistant. Your primary responsibility is to manage tasks related to company sales data, business insights, and internal operations.

**Your Available Agents (Tools):**
- **call_SalesDataAgent(question: str):** Use this to retrieve or visualize company sales data from the database. Input should be the user's specific question or request for a chart (e.g., "What were Q1 sales?", "Show a bar chart of sales by product category").

**Decision Logic:**
1.  **Analyze User Request:** Carefully read and understand the user's question or request.
2.  **Route to Best Agent:**
    * If the request involves **querying, analyzing, or visualizing sales data**, route to `SalesDataAgent`.
    * (Add more conditions here as you add more company agents, e.g., "If the request is about CRM, route to `crm_agent`.")
3.  **Provide Precise Input:** Ensure the `question` argument for `call_SalesDataAgent` accurately reflects the user's sales data query or charting request.
4.  **Handle Output:** When `SalesDataAgent` returns its output (data or chart file path), present that output directly as your final response.
5.  **Finish:** Once a task is completed and the output is presented, output 'FINISH'. If the request is not related to your domain, output 'FINISH' and indicate that you cannot handle the request.
"""

def create_company_supervisor_graph():
    """
    Creates and returns the Company Supervisor graph instance.
    This function should be called from yukta_nexus.py after agents are created.
    """
    if _llm is None:
        raise ValueError("Company Supervisor LLM not initialized. Call init_company_supervisor() first.")
    if _sales_data_agent_instance is None:
        raise ValueError("Company Supervisor: Sales Data Agent instance not provided. Ensure init_company_supervisor() is called with a valid Sales Data Agent object.")
    
    company_supervisor_graph = create_supervisor(
        model = _llm,
        agents = [_sales_data_agent_instance], # Use the *instance* passed via init_company_supervisor
        prompt = company_supervisor_prompt,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(name="company_supervisor") # No checkpointer here, yukta_nexus will handle global checkpointer
    return company_supervisor_graph