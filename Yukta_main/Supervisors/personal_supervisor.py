from langgraph_supervisor import create_supervisor

_llm = None
_RAG_agent_instance = None
_calendar_agent_instance = None

def init_personal_supervisor(llm_model, RAG_agent_obj, calendar_agent_object):
    """
    Initializes global dependencies for the Personal Supervisor.
    This function should be called once from yukta_nexus.py.
    """
    global _llm, _RAG_agent_instance, _calendar_agent_instance
    _llm = llm_model
    _RAG_agent_instance = RAG_agent_obj
    _calendar_agent_instance = calendar_agent_object
    

personal_supervisor_prompt = """
You are the Personal Supervisor within the 'Yukta' AI Assistant. Your primary responsibility is to manage tasks related to personal information, private documents, and specific knowledge bases, including scheduling and calendar management.

**Your Available Agents (Tools):**
- **call_RAG_agent(question: str):** Use this to answer questions strictly based on internal personal documents, such as FutureSmart AI's college syllabus. Input should be the user's specific question about these documents.
- **call_calendar_agent(...):** Use this for creating, searching, or deleting events in the user's Google Calendar. This agent will handle the details of calendar operations.

**Decision Logic:**
1.  **Analyze User Request:** Carefully read and understand the user's question.
2.  **Route to Best Agent:**
    * If the request is about **finding information within personal documents, especially the FutureSmart AI college syllabus**, route to `RAG_agent`.
    * If the request is about **creating, searching, updating, or deleting events in a calendar, or setting reminders related to dates/times**, route to `calendar_agent`.
    * (Add more conditions here as you add more personal agents, e.g., "If the request is about notes, route to `notes_agent`.")
3.  **Provide Precise Input:** Ensure the arguments for the chosen agent's tool are directly derived from the user's query.
4.  **Handle Output:** When the selected agent returns its answer, present that answer directly as your final response.
5.  **Finish:** Once a task is completed and the output is presented, output 'FINISH'. If the request is not related to your domain, output 'FINISH' and indicate that you cannot handle the request.
"""

def create_personal_supervisor_graph():
    """
    Creates and returns the Personal Supervisor graph instance.
    This function should be called from yukta_nexus.py after agents are created.
    """
    if _llm is None:
        raise ValueError("Personal Supervisor LLM not initialized. Call init_personal_supervisor() first.")
    if _RAG_agent_instance is None:
        raise ValueError("Personal Supervisor: Agent instances not initialized. Ensure create_agent functions are called.")
    if _calendar_agent_instance is None:
        raise ValueError("Personal Supervisor: Calendar Agent instance is not initialized.")

    personal_supervisor_graph = create_supervisor(
        model = _llm,
        agents = [_RAG_agent_instance, _calendar_agent_instance], # Use the *instances*
        prompt = personal_supervisor_prompt,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(name="personal_supervisor") # No checkpointer here, yukta_nexus will handle global checkpointer
    return personal_supervisor_graph


