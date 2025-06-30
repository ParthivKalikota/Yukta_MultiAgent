# Yukta: Your Intelligent Multi-Domain AI Personal Assistant

Yukta is a cutting-edge AI personal assistant designed to streamline your daily tasks across various domains. Built on a sophisticated **hierarchical multi-agent architecture** using LangChain and LangGraph, Yukta intelligently delegates and orchestrates complex workflows, providing specialized and context-aware assistance.

---

## 🌟 Features

Yukta is equipped with a suite of specialized agents, orchestrated by intelligent supervisors, to handle a wide range of requests:

* **Hierarchical Multi-Agent Architecture:**
    * **Yukta Prime (Meta-Supervisor):** The central intelligence that routes requests to the most appropriate specialized supervisor or orchestrates multi-step plans across domains.
    * **Communication Supervisor:** Manages external communication, content generation, and general web research.
    * **Personal Supervisor:** Handles personal information, private documents, and calendar management.
    * **Company Supervisor:** Focuses on company sales data, business insights, and internal operations.
* **Specialized Agent Capabilities:**
    * **Personal RAG Agent:** Answers questions strictly based on internal personal documents (e.g., college syllabus PDFs) using a Pinecone Vector Store.
    * **Research Agent:** Performs broad web searches for general knowledge, current events, and factual information via Tavily.
    * **LinkedIn Agent:** Generates professional and engaging LinkedIn posts with structured output.
    * **Email Agent:** Drafts and reviews professional emails with structured content and feedback.
    * **Sales Data Agent:** Queries PostgreSQL databases for sales data, performs analysis, and generates insightful charts (bar, pie) using Pandas and Matplotlib.
    * **Google Calendar Agent (NEW):** Integrates directly with Google Calendar to create, search, and delete events, and manage reminders through natural language.
* **Conversational Memory:**
    * **Short-Term Memory:** Utilizes LangGraph's checkpointer to maintain context across multi-turn conversations, enabling seamless and coherent dialogues within a session.
* **Proactive Assistance (Initial Stage):**
    * Offers contextual suggestions based on ongoing conversations.
    * Provides data-driven alerts by identifying notable trends or anomalies from agent outputs.
* **Intuitive UI:**
    * **Streamlit Web Interface:** A basic web-based chat UI built with Streamlit for an accessible and functional interaction experience during initial development and testing.
* **Robust & Secure Interactions:**
    * SQL Agent uses a query checker and strict `SELECT` only policies for safe database interactions.
    * Structured outputs via Pydantic models ensure reliable data exchange between agents and tools.

---

## 🏛️ Architecture Overview

Yukta employs a sophisticated **hierarchical multi-agent system**:

1.  **Yukta Prime (`yukta_nexus_graph`):** The top-level supervisor. It receives user queries, determines the overall intent, formsulates multi-step plans if necessary, and delegates to the appropriate specialized sub-supervisor.
2.  **Domain-Specific Supervisors:**
    * `communication_supervisor`
    * `personal_supervisor`
    * `company_supervisor`
    These supervisors manage a group of highly specialized individual agents within their domain. They interpret refined requests from Yukta Prime and orchestrate their sub-agents.
3.  **Individual Agents:** (`RAG_agent`, `research_agent`, `linkedin_agent`, `email_agent`, `SalesDataAgent`, `GoogleCalendarAgent`) These are the workers. Each agent possesses specific tools and prompts, enabling them to perform highly focused tasks (e.g., interacting with Pinecone, Tavily, PostgreSQL, or Google Calendar API).

Dependencies (LLMs, API keys, DB connections) are injected centrally from `yukta_nexus.py` down to the individual agents and supervisors, promoting modularity and testability.

---

## 🛠️ Setup and Installation

Follow these steps to get Yukta up and running on your local machine.

### Prerequisites

* Python 3.9+
* `pip` (Python package installer)
* **OpenAI API Key:** For accessing GPT-4o models.
* **Tavily API Key:** For web search capabilities.
* **NVIDIA API Key:** For NVIDIA embeddings (used in RAG).
* **Pinecone API Key:** For vector database functionalities.
* **Pinecone Indexes:** Ensure you have the `rag-documents-index` created in your Pinecone account.
* **PostgreSQL Database:** A running PostgreSQL instance with a `sales` table populated with your data.
* **Google Calendar API Credentials:**
    * Go to [Google Cloud Console](https://console.cloud.google.com/).
    * Create a new project.
    * Enable the "Google Calendar API".
    * Go to "APIs & Services" > "Credentials".
    * Click "Create Credentials" > "OAuth client ID" > "Desktop app".
    * Download the `credentials.json` file and place it in your project's root directory.

### 1. Clone the Repository

```bash
git clone [YOUR_GITHUB_REPO_LINK]
cd [your_project_root_directory]
