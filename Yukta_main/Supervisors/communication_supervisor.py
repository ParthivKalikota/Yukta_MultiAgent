from langgraph_supervisor import create_supervisor

_llm = None
_research_agent_instance = None
_email_agent_instance = None
_linkedin_agent_instance = None

def init_communication_supervisor(llm_model, research_agent_obj, email_agent_obj, linkedin_agent_obj):
    """
    Initializes global dependencies for the Communication Supervisor.
    This function should be called once from yukta_nexus.py.
    """
    global _llm, _research_agent_instance, _email_agent_instance, _linkedin_agent_instance
    _llm = llm_model
    _research_agent_instance = research_agent_obj
    _email_agent_instance = email_agent_obj
    _linkedin_agent_instance = linkedin_agent_obj

communication_supervisor_prompt = """
You are the Communication Supervisor within the 'Yukta' AI Assistant. Your primary responsibility is to manage tasks related to external communication, content generation, and general web research.

**Your Available Agents (Tools):**
- **call_research_agent(query: str):** Use this to perform broad web searches for general knowledge, current events, or factual information. Input should be a precise search query.
- **call_email_agent(request: str, applicant_name: str = None, applicant_phone: str = None, applicant_email: str = None):** Use this for drafting or reviewing emails. The 'request' should clearly state the email's purpose. If drafting, provide `applicant_name`, `applicant_phone`, and `applicant_email` if available.
- **call_linkedin_agent(user_input: str):** Use this to generate professional LinkedIn posts based on the provided 'user_input' (the desired content for the post).

**Decision Logic:**
1.  **Analyze User Request:** Carefully read and understand the user's request.
2.  **Route to Best Agent:**
    * If the request clearly involves **general web search, factual lookup, or current events**, route to `research_agent`.
    * If the request is about **writing or reviewing an email**, route to `email_agent`.
    * If the request is about **generating a LinkedIn post**, route to `linkedin_agent`.
3.  **Provide Precise Input:** Ensure the arguments passed to the chosen agent's tool (`query`, `request`, `user_input`, and applicant details) are extracted accurately and completely from the user's original query.
4.  **Handle Output:** When the selected agent returns its final output, present that output directly as your own final response.
5.  **Finish:** Once a task is completed and the output is presented, output 'FINISH'. If no suitable agent is found, output 'FINISH' and indicate that you cannot handle the request.
"""

def create_communication_supervisor_graph():
    """
    Creates and returns the Communication Supervisor graph instance.
    This function should be called from yukta_nexus.py after agents are created.
    """
    if _llm is None:
        raise ValueError("Communication Supervisor LLM not initialized. Call init_communication_supervisor() first.")
    if any(a is None for a in [_research_agent_instance, _email_agent_instance, _linkedin_agent_instance]):
        raise ValueError("Communication Supervisor: Agent instances not initialized. Ensure create_agent functions are called.")
    
    communication_supervisor_graph = create_supervisor(
        model = _llm,
        agents = [_research_agent_instance, _email_agent_instance, _linkedin_agent_instance], # Use the *instances*
        prompt = communication_supervisor_prompt,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(name="communication_supervisor") # No checkpointer here, yukta_nexus will handle global checkpointer
    return communication_supervisor_graph

