import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

class LinkedInPost(BaseModel):
    """
    Structured output for a LinkedIn post, designed for professionalism and engagement.
    """
    hook: str = Field(
        description="A compelling, short opening statement (1-2 sentences) "
                    "designed to grab immediate attention and entice the reader "
                    "to click 'See more' or continue reading. It should pose a question, "
                    "state a bold claim, or highlight a surprising fact related to the post's topic."
    )
    body_content: str = Field(
        description="The main body of the LinkedIn post. This should be concise (150-300 words), "
                    "deliver the core message, share insights, provide value, and ideally include "
                    "a call to action or a thought-provoking question at the end. "
                    "It should be well-structured with short paragraphs or bullet points for readability."
    )
    hashtags: List[str] = Field(
        description="A list of 3-7 relevant and popular hashtags (e.g., '#AI', '#TechInnovation', '#Productivity'). "
                    "These increase the post's discoverability and reach a wider, relevant audience. "
                    "Do NOT include the '#' symbol in the individual list items; it will be added during formatting."
    )
    call_to_action: Optional[str] = Field(
        default=None,
        description="An optional clear and concise call to action at the end of the body content, "
                    "such as 'What are your thoughts?' or 'Learn more here!' or 'Connect with me to discuss!'"
    )


_LinkedIn_llm = None
_linkedin_parser = None
linkedin_post_prompt = None
linkedin_post_chain = None

def init_linkedin_agent(LinkedIn_llm):
    global _LinkedIn_llm, _linkedin_parser
    global linkedin_post_prompt, linkedin_post_chain

    _LinkedIn_llm = LinkedIn_llm
    _linkedin_parser = PydanticOutputParser(pydantic_object=LinkedInPost)

    linkedin_post_prompt = PromptTemplate(
        template="""You are an expert in preparing/creating highly engaging and professional LinkedIn posts.
        Your task is to take the provided information and user requests to generate a LinkedIn post.
        The post MUST adhere to the following JSON structure.
        Ensure all fields are accurately and creatively filled based on the input.

        {format_instructions}

        Here is the User's Information and Request:
        User Request: {user_input}

        ---
        Guidelines for Post Generation:
        - **Hook:** Make it irresistible, encouraging immediate engagement.
        - **Body Content:** Provide genuine value, insights, or a compelling narrative. Break it into short paragraphs or use bullet points. Keep it professional. Max 300 words.
        - **Hashtags:** Generate 3-7 relevant and trending hashtags. Do NOT include '#' symbol in the list items.
        - **Call to Action (Optional):** Include a subtle call to action if appropriate, encouraging comments or further engagement.
        """,
        input_variables=['user_input'],
        partial_variables={'format_instructions': _linkedin_parser.get_format_instructions()}
    )

    linkedin_post_chain = linkedin_post_prompt | _LinkedIn_llm | _linkedin_parser

@tool
def generate_linkedin_post(user_input: str) -> LinkedInPost:
    """
    Generates a structured LinkedIn post based on user-provided content.
    """
    print("\n--- INSIDE LINKEDIN POST GENERATOR TOOL ---")
    try:
        generated_post = linkedin_post_chain.invoke({'user_input': user_input})
        print("LinkedIn Post generated successfully.")
        return generated_post
    except Exception as e:
        print(f"Error generating LinkedIn post: {e}")
        return LinkedInPost(
            hook="Error generating post",
            body_content=f"An error occurred during post generation: {e}",
            hashtags=["Error"],
            call_to_action="Please try again or rephrase your request."
        )

@tool
def format_linkedin_post_for_display(post_obj: LinkedInPost) -> str:
    """
    Formats a LinkedInPost Pydantic object into a human-readable string,
    suitable for displaying or copying directly to LinkedIn.
    """
    formatted_post = ""

    # Add the hook
    formatted_post += post_obj.hook
    formatted_post += "\n\n" # Two line breaks for LinkedIn readability

    # Add the body content
    formatted_post += post_obj.body_content
    formatted_post += "\n\n" # Two line breaks

    # Add the call to action if present
    if post_obj.call_to_action:
        formatted_post += post_obj.call_to_action
        formatted_post += "\n\n" # Two line breaks if CTA exists

    # Add hashtags, ensuring they start with '#'
    if post_obj.hashtags:
        formatted_hashtags = " ".join([f"#{tag.strip()}" for tag in post_obj.hashtags if tag.strip()])
        formatted_post += formatted_hashtags

    return formatted_post.strip()

linkedin_agent_prompt = """You are a specialized LinkedIn Post Creation and Formatting Agent.
Your primary goal is to generate engaging LinkedIn posts and present them in a perfectly formatted, human-readable string.
You will receive a request that details what kind of LinkedIn post needs to be created.
Follow the workflow instructions precisely.

**Here are your available tools:**
1.  `generate_linkedin_post_tool(user_input: str)`: Use this tool first when you need to create a LinkedIn post. Provide the user's detailed request as input to this tool. This tool will return a structured LinkedInPost object.
2.  `format_linkedin_post_for_display_tool(post_obj: LinkedInPost)`: Use this tool *immediately after* `generate_linkedin_post_tool` has successfully returned a LinkedInPost object. Provide the LinkedInPost object (the output from `generate_linkedin_post_tool`) as input to this tool. This tool will return the final, nicely formatted string for the post.

**Workflow Instructions:**
-   **Step 1: Generate Post.** Begin by extracting the core request for the LinkedIn post from the request you received. Then, use the `generate_linkedin_post_tool` with that extracted request.
-   **Step 2: Format Post.** Once you receive the structured LinkedInPost object from `generate_linkedin_post_tool` (which will appear as a tool output in your scratchpad), immediately use the `format_linkedin_post_for_display_tool` with that object as input.
-   **Step 3: Present Final Result.** After the `format_linkedin_post_for_display_tool` returns the formatted string, your task is complete. Present this formatted string as your final answer to the supervisor.
-   Do NOT include any additional conversational text or explanations in your final output, ONLY the formatted LinkedIn post string."""

def create_linkedin_agent():
    linkedin_agent = create_react_agent(
    model = _LinkedIn_llm,
    tools = [generate_linkedin_post, format_linkedin_post_for_display],
    prompt = linkedin_agent_prompt,
    name= 'linkedin_agent'
    )
    return linkedin_agent
