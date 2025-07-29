import streamlit as st
import os
import hashlib
import pickle
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain import hub
# ENV setup
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]

st.set_page_config(page_title="Chat with Cached PDFs", layout="wide")
st.title("📄 Chat with Multiple PDFs + Cached FAISS DB")

# Constants
DB_PATH = "vector_store"
HASH_PATH = "doc_hash.pkl"

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if st.sidebar.button("🔄 Reset Chat"):
    st.session_state.clear()
    st.rerun()


# Helper: calculate hash of all PDFs
def compute_files_hash(files):
    sha = hashlib.sha256()
    for file in files:
        content = file.read()
        sha.update(content)
        file.seek(0)  # Reset pointer
    return sha.hexdigest()


# Load previous hash
def load_prev_hash():
    if os.path.exists(HASH_PATH):
        with open(HASH_PATH, "rb") as f:
            return pickle.load(f)
    return None


# Save current hash
def save_current_hash(hash_value):
    with open(HASH_PATH, "wb") as f:
        pickle.dump(hash_value, f)

#-----------
wiki_tool = Tool(
        name="wikipedia_search",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        description="Search Wikipedia for general facts."
    )

llm = ChatOpenAI(temperature=0)
openai_tool = Tool(
        name="openai_search",
        func=lambda q: llm.predict(q),
        description="Fallback to OpenAI LLM if other tools can't help."
    )

    # Agent setup
tools = [wiki_tool, openai_tool]
if st.session_state.agent_executor is None:
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    st.session_state.agent_executor = executor
    st.info("🧠 Assistant is ready! You can start chatting.")

#----------




# File upload
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    current_hash = compute_files_hash(uploaded_files)
    previous_hash = load_prev_hash()

    if os.path.exists(DB_PATH) and current_hash == previous_hash:
        # Load cached FAISS DB
        vectordb = FAISS.load_local(DB_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        st.success("Loaded cached FAISS DB ✅")
    else:
        # Build new vector store
        st.info("New files detected. Rebuilding FAISS DB...")
        all_docs = []

        for file in uploaded_files:
            with open(file.name, "wb") as f:
                f.write(file.read())

            loader = PyPDFLoader(file.name)
            docs = loader.load()
            all_docs.extend(docs)

        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(all_docs, embeddings)
        vectordb.save_local(DB_PATH)

        save_current_hash(current_hash)
        st.success("New FAISS DB built and cached ✅")

    # Create retriever
    retriever = vectordb.as_retriever()

    # Define tools
    def search_pdf(query: str):
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs[:2]) if docs else "No relevant info found."

    pdf_tool = Tool(
        name="pdf_search",
        func=search_pdf,
        description="Search the uploaded PDFs for relevant information."
    )
    tools.insert(0, pdf_tool) 
    # wiki_tool = Tool(
    #     name="wikipedia_search",
    #     func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    #     description="Search Wikipedia for general facts."
    # )

    # llm = ChatOpenAI(temperature=0)
    # openai_tool = Tool(
    #     name="openai_search",
    #     func=lambda q: llm.predict(q),
    #     description="Fallback to OpenAI LLM if other tools can't help."
    # )

    # # Agent setup
    # tools = [pdf_tool, wiki_tool, openai_tool]
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful assistant with access to tools like PDF search, Wikipedia, and OpenAI. Use them smartly."),
    #     ("user", "{input}")
    # ])
    prompt = hub.pull("hwchase17/openai-functions-agent")
    print(prompt)
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    st.session_state.agent_executor = executor
    #st.chat_message("assistant").write("You can start asking questions!")

# User input
user_question = st.chat_input("Ask your question")

if user_question and st.session_state.agent_executor:
   # st.chat_message("user").write(user_question)
    result = st.session_state.agent_executor.invoke({"input": user_question})
    answer = result["output"]

    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    #st.chat_message("assistant").write(answer)

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
