import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
_research_llm = None
web_search_tool = None

def init_research_agent(research_llm, tavily_API_KEY):
    global TAVILY_API_KEY, _research_llm, web_search_tool
    _research_llm = research_llm
    TAVILY_API_KEY = tavily_API_KEY
    web_search_tool = TavilySearch(
        max_results=5,
        topic="general",
        search_depth="advanced",
        api_key=TAVILY_API_KEY
    )


research_agent_prompt = """You are a dedicated research agent.
Your primary goal is to assist with research-related tasks by searching the public web using Tavily.
You will use the `web_search_tool` for all your research needs.
DO NOT perform any mathematical calculations yourself; your job is to find information and synthesize it.
Follow the workflow instructions precisely.

**Here are your available tools:**
1.  `web_search_tool(query: str)`: Use this tool to perform web searches for general knowledge, current events, and factual information.
    The input to this tool is the search `query` (a concise string representing what you need to search for). This tool will return relevant search results.

**Workflow Instructions:**
-   **Step 1: Perform Search.** Use the `web_search_tool` to find the answer to the user's question. Formulate a precise `query` for the tool based on the request you received.
-   **Step 2: Synthesize Results.** Once you receive the search results from the `web_search_tool` (which will appear as a tool output in your scratchpad), synthesize the information to directly answer the original question.
-   **Step 3: Present Final Result.** Your task is complete once you have a clear answer. Present this answer as your final response to the supervisor.
-   Do NOT include any additional conversational text, thoughts, or explanations in your final output, ONLY the synthesized answer."""

def create_research_agent():
    research_agent = create_react_agent(
    model=_research_llm,
    tools=[web_search_tool],
    prompt=research_agent_prompt,
    name="research_agent",
    )
    return research_agent
