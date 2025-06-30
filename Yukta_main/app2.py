# app.py

import streamlit as st
import uuid # For generating unique session IDs
import os
from dotenv import load_dotenv

# Import the main graph initialization function from yukta_nexus.py
from yukta_nexus import initialize_yukta_graph
from langchain_core.messages import AIMessage, HumanMessage

# --- Configuration (Load Environment Variables) ---
load_dotenv()

# --- Retrieve Environment Variables ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-documents-index")

PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DBNAME = os.getenv("PG_DBNAME")
DATABASE_URI = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"


# --- LLM Config for initialization ---
llm_config = {
    'default_model': 'gpt-4o',
    'rag_model': 'gpt-4o',
    'research_model': 'gpt-4o',
    'linkedin_model': 'gpt-4o',
    'linkedin_temp': 0.8,
    'email_writer_model': 'gpt-4o',
    'email_writer_temp': 0.7,
    'email_reviewer_model': 'gpt-4o',
    'sales_model': 'gpt-4o',
    'yukta_nexus_model': 'gpt-4o',
    'embedding_model': "nvidia/llama-3.2-nv-embedqa-1b-v2",
    'calendar_model' : 'gpt-4o'
}

# --- API Keys Config ---
api_keys = {
    'OPENAI_API_KEY': OPENAI_API_KEY,
    'TAVILY_API_KEY': TAVILY_API_KEY,
    'NVIDIA_API_KEY': NVIDIA_API_KEY,
    'PINECONE_API_KEY': PINECONE_API_KEY
}


# --- Initialize Yukta and get the compiled graph and memory saver ---
@st.cache_resource(show_spinner="Starting Yukta AI Assistant. This might take a moment...")
def cached_initialize_yukta_graph():
    """Initializes the entire Yukta graph and its components, caching the result."""
    yukta_graph, checkpointer = initialize_yukta_graph(
        llm_config,
        api_keys,
        DATABASE_URI,
        './TestData',
        PINECONE_INDEX_NAME
    )
    st.success("Yukta AI Assistant Core Initialized!")
    return yukta_graph, checkpointer

# --- Get the initialized Yukta graph and checkpointer instance ---
yukta_nexus_graph, session_memory_saver = cached_initialize_yukta_graph()


# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="Yukta AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded" # Optional: Start sidebar expanded
)

# Load custom CSS
st.markdown("""
    <style>
    /* Adjust Streamlit's default markdown to use your theme text color */
    .stMarkdown {
        color: var(--text-color);
    }
    /* Specific styling for chat containers */
    .st-chat-message-container > div:first-child { /* Targets the avatar column */
        align-self: flex-end;
    }
    .st-chat-message-container > div:last-child { /* Targets the message bubble column */
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    .st-chat-message-container.user > div:last-child {
        align-items: flex-end;
    }
    .st-chat-message-container.assistant > div:last-child {
        align-items: flex-start;
    }

    /* Apply custom CSS from file */
    """ + open("style.css").read() + """
    </style>
""", unsafe_allow_html=True)


# --- UI Elements ---
# Header
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.title("🤝 Yukta")
    st.markdown("<h3 style='text-align: center; color: var(--primary-color);'>Your personalized multi-domain AI assistant.</h3>", unsafe_allow_html=True)
    st.markdown("---") # Visual separator

# Chat history display area
chat_history_container = st.container(height=500, border=False) # Fixed height container for chat, no border

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []
# Initialize a unique thread_id for this Streamlit session for LangGraph's checkpointer
if "thread_id" not in st.session_state: # Corrected syntax: `not in`
    st.session_state.thread_id = str(uuid.uuid4()) # Generates a new unique ID for each new browser session

# Display chat messages from history on app rerun
with chat_history_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Special handling for chart images based on their expected response format
            if message["role"] == "assistant" and "Chart generated successfully:" in message["content"]:
                image_path = message["content"].replace("Chart generated successfully:", "").strip()
                if os.path.exists(image_path):
                    st.image(image_path, caption="Generated Chart", use_column_width=True)
                    st.markdown(f"Here is your chart: `{os.path.basename(image_path)}`")
                else:
                    st.markdown(f"Yukta generated a chart, but the image file was not found at `{image_path}`. Raw response: {message['content']}")
            else:
                st.markdown(message["content"])


# Accept user input at the bottom of the chat interface
if prompt := st.chat_input("How can Yukta help you today?"):
    # Add user's new message to Streamlit's chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_history_container.chat_message("user"): # Display in the fixed height container
        st.markdown(prompt)

    # Prepare LangGraph configuration with the unique thread_id for conversational memory
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with chat_history_container.chat_message("assistant"): # Display in the fixed height container
        message_placeholder = st.empty() # Create an empty placeholder to update with response
        
        try:
            # Use yukta_nexus_graph.invoke() method to get the final response.
            # Using invoke() is simpler for UI as it returns the complete final state,
            # unlike stream() which provides granular intermediate updates.
            with st.spinner("Yukta is thinking..."): # Show a spinner while processing
                final_state = yukta_nexus_graph.invoke(
                    {"messages": [HumanMessage(content=prompt)]}, # Input is a list containing the user's HumanMessage
                    config=config # Pass the config to enable short-term memory
                )

            # Extract the final AI message from the complete state after invocation
            final_ai_message = None
            if "messages" in final_state and final_state["messages"]:
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage):
                        final_ai_message = msg
                        break
            
            if final_ai_message:
                full_response = final_ai_message.content
                
                # Special check: If the response indicates a chart was generated, display the image
                if "Chart generated successfully:" in full_response:
                    image_path_str = full_response.replace("Chart generated successfully:", "").strip()
                    if os.path.exists(image_path_str):
                        st.image(image_path_str, caption="Generated Chart", use_column_width=True)
                        message_placeholder.markdown(f"Here is your chart: `{os.path.basename(image_path_str)}`")
                    else:
                        message_placeholder.markdown(f"Yukta generated a chart, but the image file was not found at `{image_path_str}`. Raw response: {full_response}")
                else:
                    message_placeholder.markdown(full_response) # Display text response

                # Append the final AI response to Streamlit's session history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                # Fallback if no clear final AI message is found in the state
                message_placeholder.markdown("Yukta could not generate a clear response for this query.")
                st.session_state.messages.append({"role": "assistant", "content": "Yukta could not generate a clear response for this query."})
                
        except Exception as e:
            st.error(f"An internal error occurred: {e}. Please check your API keys, database, and Pinecone connections.")
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}. Please try again."})


if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4()) # Generate new thread_id for a fresh start
    st.rerun() # CORRECTED: Use st.rerun()