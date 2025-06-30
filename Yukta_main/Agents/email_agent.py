import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

_llm = None
_email_writer_llm = None
_email_reviewer_llm = None

_email_writer_parser = None
_email_reviewer_parser = None

email_writer_prompt = None
email_reviewer_prompt = None

class EmailContent(BaseModel):
    """Structured output for an email, including its subject, body, and recipient details."""
    recipient_name: str = Field(description="The name of the person the email is addressed to (e.g., 'John Doe', 'Hiring Manager').")
    recipient_greeting: str = Field(description="The opening greeting of the email (e.g., 'Dear Mr. Smith,', 'Hello Team,').")
    subject: str = Field(description="The concise subject line of the email.")
    body: str = Field(description="The main content of the email, formatted as a clear and professional message.")
    applicant_name: str = Field(description="The full name of the sender/applicant.")
    applicant_phone: str = Field(description="The phone number of the sender/applicant.")
    applicant_email: str = Field(description="The email address of the sender/applicant.")
    closing: str = Field(description="The closing phrase of the email (e.g., 'Sincerely,', 'Regards,').")


class EmailReviewFeedback(BaseModel):
    """Structured feedback for an email review."""
    approved: bool = Field(description="True if the email is approved with no significant changes needed, False otherwise.")
    suggestions: str = Field(description="Detailed suggestions for improvement if not approved, or 'None' if approved.")
    revised_subject: str = Field(description="The revised subject line if changes are suggested, otherwise same as original.")
    revised_body: str = Field(description="The revised email body if changes are suggested, otherwise same as original.")

def init_email_agent(llm_model, email_writer_llm, email_reviewer_llm):
    global _llm, _email_writer_llm, _email_reviewer_llm
    global _email_writer_parser, _email_reviewer_parser
    global email_reviewer_prompt, email_writer_prompt
    _llm = llm_model
    _email_writer_llm = email_writer_llm
    _email_reviewer_llm = email_reviewer_llm

    _email_writer_parser = PydanticOutputParser(pydantic_object=EmailContent)
    _email_reviewer_parser = PydanticOutputParser(pydantic_object=EmailReviewFeedback)

    email_writer_prompt = PromptTemplate(
        template = """You are an expert at writing professional emails.
        Your task is to write a complete email based on the user's request.
        The email MUST adhere to the following JSON structure.
        Ensure all fields are filled accurately based on the request.
        If a specific detail is not provided, use a reasonable placeholder (e.g., "Hiring Manager" for recipient, or "N/A" for phone if not given).

        {format_instructions}

        Here is the user's request:
        Request: {user_request}

        Here are the applicant details that must be used at the end of the email:
        Applicant Name: {applicant_name}
        Applicant Phone: {applicant_phone}
        Applicant Email: {applicant_email}

        Ensure the email's body is well-structured and professional.
        """,
        input_variables=['user_request', 'applicant_name', 'applicant_phone', 'applicant_email'],
        partial_variables={"format_instructions": _email_writer_parser.get_format_instructions()}
    )

    email_reviewer_prompt = PromptTemplate(
    template="""You are a professional email reviewer.
    Your task is to analyze the provided email content for clarity, conciseness, grammar, tone, and professionalism.
    Provide constructive feedback and suggest specific revisions if needed.
    You MUST output your review in the following JSON structure.

    {format_instructions}

    Here is the email content to review:
    Recipient: {recipient_name}
    Greeting: {recipient_greeting}
    Subject: {subject}
    Body:
    {body}
    Closing: {closing}
    Applicant: {applicant_name} ({applicant_email}, {applicant_phone})

    Analyze the email for:
    - Clarity: Is the message easy to understand?
    - Conciseness: Is there any unnecessary jargon or lengthy phrasing?
    - Grammar & Spelling: Any errors?
    - Tone: Is it appropriate (e.g., professional, polite, firm)?
    - Completeness: Does it address all aspects of the original request?
    - Overall Professionalism.

    Provide your specific suggestions and populate the revised fields.
    """,
    input_variables=[
        'recipient_name', 'recipient_greeting', 'subject', 'body', 'closing',
        'applicant_name', 'applicant_email', 'applicant_phone'
    ],
    partial_variables={"format_instructions": _email_reviewer_parser.get_format_instructions()}
    )

@tool
def write_email_tool(user_request: str,
    applicant_name: str,
    applicant_phone: str,
    applicant_email: str) -> EmailContent:
    """
    Writes a professional email based on the user's request and applicant details.
    Outputs the email content in a structured Pydantic object.

    Args:
        user_request (str): The user's detailed request for the email, including purpose, company, role, etc.
        applicant_name (str): The full name of the sender.
        applicant_phone (str): The phone number of the sender.
        applicant_email (str): The email address of the sender.
    """
    print("INSIDE EMAIL WRITER TOOL")
    try:
        email_chain = email_writer_prompt | _email_writer_llm | _email_writer_parser
        generated_email_obj = email_chain.invoke({
            'user_request': user_request,
            'applicant_name': applicant_name,
            'applicant_phone': applicant_phone,
            'applicant_email': applicant_email
        })
        return generated_email_obj
    except Exception as e:
        print(f"Error in write_email_tool: {e}")
        # Return an EmailContent object with error details for consistent type
        return EmailContent(
            recipient_name="Recipient", # Placeholder
            recipient_greeting="Dear Sir/Madam,", # Placeholder
            subject="Error: Email Generation Failed",
            body=f"An error occurred while drafting the email: {e}",
            applicant_name=applicant_name,
            applicant_phone=applicant_phone,
            applicant_email=applicant_email,
            closing="Regards," # Placeholder
        )




@tool
def review_email_tool(email_content: EmailContent) -> EmailReviewFeedback:
    """
    Reviews a structured email content object for professionalism and provides feedback.

    Args:
        email_content (EmailContent): The structured email content generated by the EmailWriterAgent.
    """
    print("INSIDE EMAIL REVIEWER TOOL")
    try:
        review_chain = email_reviewer_prompt | _email_reviewer_llm | _email_reviewer_parser

        # Pass all relevant fields from the EmailContent object to the prompt
        review_feedback_obj = review_chain.invoke({
            'recipient_name': email_content.recipient_name,
            'recipient_greeting': email_content.recipient_greeting,
            'subject': email_content.subject,
            'body': email_content.body,
            'closing': email_content.closing,
            'applicant_name': email_content.applicant_name,
            'applicant_email': email_content.applicant_email,
            'applicant_phone': email_content.applicant_phone
        })
        return review_feedback_obj
    except Exception as e:
        print(f"Error in review_email_tool: {e}")
        # Return an EmailReviewFeedback object with error details for consistent type
        return EmailReviewFeedback(
            approved=False,
            suggestions=f"An error occurred during email review: {e}",
            revised_subject=email_content.subject, # Keep original
            revised_body=email_content.body # Keep original
        )
    
email_agent_prompt = """You are a dedicated Email Management Agent. Your task is to handle all email-related requests, including drafting and reviewing emails.
You have access to `write_email_tool` and `review_email_tool`.
Follow the workflow instructions precisely.

**Here are your available tools:**
1.  `write_email_tool(user_request: str, applicant_name: str, applicant_phone: str, applicant_email: str)`: Use this tool to draft a new email.
    Provide the full email request and applicant details as input. This tool returns a structured EmailContent object.
2.  `review_email_tool(email_content: EmailContent)`: Use this tool to review an existing structured email.
    Provide the EmailContent object as input. This tool returns structured EmailReviewFeedback.

**Workflow Instructions:**
-   **If the request is to draft an email:**
    -   **Step A:** Use the `write_email_tool`. Extract `user_request`, `applicant_name`, `applicant_phone`, `applicant_email` from the request you received.
    -   **Step B (Optional/Conditional):** After receiving the `EmailContent` object, if the user explicitly requested a review or if review is standard, use `review_email_tool` with the drafted `EmailContent` object.
    -   **Step C: Final Output.** Present the output of the final tool (either `EmailContent` from writing or `EmailReviewFeedback` from reviewing) as your final response to the supervisor.
-   **If the request is to review an email:**
    -   **Step A:** Extract the `EmailContent` object from the request (if the supervisor passes it in a structured way).
    -   **Step B:** Use the `review_email_tool` with the `EmailContent` object.
    -   **Step C: Final Output.** Present the `EmailReviewFeedback` object as your final response to the supervisor.
-   Do NOT add extra conversational text to your final output."""

def create_email_agent():
    """
    Creates and returns the Email agent instance.
    This function should be called from yukta_nexus.py after init_email_agent().
    """
    if _llm is None or _email_writer_llm is None or _email_reviewer_llm is None:
        raise ValueError("Email Agent LLM dependencies not initialized. Call init_email_agent() first.")
    email_agent = create_react_agent(
    model = _llm, 
    tools=[write_email_tool, review_email_tool],
    prompt=email_agent_prompt,
    name = 'email_agent'
    )
    return email_agent