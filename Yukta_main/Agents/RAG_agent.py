import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from pinecone import Pinecone, ServerlessSpec

_RAG_llm = None
_embedding = None
_PINECONE_INDEX_NAME = None
_parser = None
vector_store = None

def init_rag_agent(RAG_llm, embedding, pinecone_rag_index_name, parser):
    global _RAG_llm, _embedding, _PINECONE_INDEX_NAME, _parser, vector_store

    _RAG_llm = RAG_llm
    _embedding = embedding
    _PINECONE_INDEX_NAME = pinecone_rag_index_name
    _parser = parser
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(_PINECONE_INDEX_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=_embedding)

# loader = DirectoryLoader(path='./TestData',glob='**/*.pdf', loader_cls=PyPDFLoader)
# docs = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300) # Adjusted for balance
# chunks = splitter.split_documents(docs)
# vector_store = None

# if chunks:
#         vector_store = PineconeVectorStore.from_documents(
#                 chunks,
#                 embedding,
#                 index_name=PINECONE_INDEX_NAME
#         )
#         print(f"Successfully Loaded {len(chunks)} chunks into Pinecone.")
# else:
#         print("Warning: No documents or chunks were processed for the RAG System. RAG Functionalities might be limited")



@tool
def retriever_tool(question: str):
    """Tool to Retrieve Semantically Similar documents to answer User Questions related to FutureSmart AI"""
    print("INSIDE RETRIEVER NODE")
    if vector_store is None:
        return "RAG system is not initialized. Please ensure documents are loaded correctly."
    retriever = MultiQueryRetriever.from_llm(
          retriever=vector_store.as_retriever(search_kwargs={'k': 4}),
          llm=_RAG_llm
    )
    prompt = PromptTemplate(
          template="""You are an AI assistant. Your sole purpose is to answer questions based *strictly and exclusively* on the provided document excerpts (Context).

          Context:
          {context_text}

          Question: {question}

          Based *only* on the context above, provide a concise and factual answer to the question.
          If the context does not contain the information to answer the question, you MUST state: "The provided document excerpts do not contain sufficient information to answer this question."
          Do NOT use any external knowledge, make assumptions, or infer information beyond what is explicitly stated in the context.
          Do NOT engage in general conversation or answer off-topic questions. If the question is not about the document's content, state that you can only answer questions based on the provided document.
          Answer:""",
          input_variables=['context_text', 'question']
        )
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    chain = prompt | _RAG_llm | _parser
    generated_answer = chain.invoke({'context_text': context_text, 'question': question})
    return generated_answer

rag_agent_prompt = """You are a specialized RAG (Retrieval Augmented Generation) agent for FutureSmart AI.
            Your primary goal is to answer user questions *strictly* based on the provided document excerpts related to FutureSmart AI's college syllabus.
            You will use the `retriever_tool` to find relevant information.
            Do NOT use any external knowledge. If the provided context does not contain the answer, state that explicitly.
            Follow the workflow instructions precisely.

            **Here are your available tools:**
            1.  `retriever_tool(question: str)`: Use this tool to search the internal knowledge base for information related to FutureSmart AI's college syllabus.
                The input to this tool is the user's specific question about the syllabus. This tool will return a concise and factual answer derived from the documents, or state if the information is not available.

            **Workflow Instructions:**
            -   **Step 1: Retrieve Information.** Use the `retriever_tool` to find the answer to the user's question. Formulate the `question` for the tool based on the core request you received.
            -   **Step 2: Present Final Result.** Once you receive the answer from the `retriever_tool` (which will appear as a tool output in your scratchpad), your task is complete. Present this answer as your final response to the supervisor.
            -   Do NOT include any additional conversational text or explanations in your final output, ONLY the answer from the `retriever_tool`."""

def create_rag_agent():
    RAG_agent = create_react_agent(
        model = _RAG_llm,
        tools = [retriever_tool],
        prompt = rag_agent_prompt,
        name = 'RAG_agent'
    )
    return RAG_agent
