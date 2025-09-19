from langchain_text_splitters import RecursiveCharacterTextSplitter
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
import pdfplumber
from langchain_core.documents import Document
# ENV setup
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.set_page_config(page_title="Chat with Cached PDFs", layout="wide")
st.title("MFI 360 Help Desk")

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
tools = [ openai_tool]
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
    # Save the first uploaded file to disk
    with open(uploaded_files[0].name, "wb") as f:
        f.write(uploaded_files[0].read())
    uploaded_files[0].seek(0)  # Reset pointer for future use
    loader = PyPDFLoader(uploaded_files[0].name)
    st.session_state.pages = loader.load()
    if   os.path.exists(DB_PATH) and current_hash == previous_hash:
        # Load cached FAISS DB
        st.session_state.vectordb = FAISS.load_local(DB_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        st.success("Loaded cached FAISS DB ✅")
    else:
        all_docs = []

        for file in uploaded_files:
            # Save uploaded file to disk
            with open(file.name, "wb") as f:
                f.write(file.read())

            # 🧠 Load unstructured text
            loader = PyPDFLoader(file.name)
            text_docs = loader.load()

            # Add metadata to each doc
            for page in text_docs:
                lines = page.page_content.splitlines()
                for line in lines:
                    if ":" in line:
                        key, value = map(str.strip, line.split(":", 1))
                        if key in ["Asset Manager", "Issuer", "Trustee"]:
                            all_docs.append(Document(
                                page_content=f"{key}: {value}",
                                metadata={"section": "kv_block", "key": key}
                            ))

    # 2. Extract and chunk full body content
        splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", ":", " "]
        )

        for page in text_docs:
            chunks = splitter.split_text(page.page_content)
            for chunk in chunks:
                all_docs.append(Document(
                    page_content=chunk,
                    metadata={"section": "body"}
                ))

        embeddings = OpenAIEmbeddings()
        st.session_state.vectordb = FAISS.from_documents(all_docs, embeddings)
        st.session_state.vectordb.save_local(DB_PATH)

        save_current_hash(current_hash)
        st.success("New FAISS DB built and cached ✅")

    # Create retriever
    if "retriever" not in st.session_state:
        st.session_state.retriever = st.session_state.vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    def search_pdf(query: str):
        # # Step 1: Vector search
        # docs = st.session_state.vectordb.similarity_search_with_score(query, k=8)
        # sorted_docs = sorted(docs, key=lambda x: x[1])
        # top_docs = [doc for doc, score in sorted_docs[:4]]

        # context = "\n\n".join(doc.page_content for doc in top_docs)

        # # Step 2: Fallback if not enough content found
        # if not context or len(context.strip()) < 30:
        #     # Try keyword-based line scan (for structured values)
        #     for page in pages:
        #         for line in page.page_content.splitlines():
        #             if any(k in line for k in ["Asset Manager", "Issuer", "Trustee"]):
        #                 context += "\n" + line

        # # Step 3: FINAL fallback — paragraph contains any keyword in query
        # if not context or len(context.strip()) < 30:
        #     for page in pages:
        #         if query.lower() in page.page_content.lower():
        #             context += "\n\n" + page.page_content

        # return context if context.strip() else "No relevant data found."

        #------------- new code
        keywords = ["Asset Manager", "Issuer", "Trustee"]
        vectordb = st.session_state.vectordb
        pages = st.session_state.pages

        # Step 1: Try exact match on known KV blocks
        if any(kw.lower() in query.lower() for kw in keywords):
            for kw in keywords:
                if kw.lower() in query.lower():
                    results = vectordb.similarity_search(query, k=5, filter={"section": "kv_block", "key": kw})
                    if results:
                        return "\n\n".join(doc.page_content for doc in results)

        # Step 2: Search in general paragraph text
        results = vectordb.similarity_search(query, k=5, filter={"section": "body"})
        if results:
            return "\n\n".join(doc.page_content for doc in results)

        # Step 3: Keyword fallback from line-by-line scan
        context = ""
        for page in pages:
            for line in page.page_content.splitlines():
                if any(k in line for k in keywords):
                    context += "\n" + line

        # Step 4: Fallback full page match
        if not context.strip():
            for page in pages:
                if query.lower() in page.page_content.lower():
                    context += "\n\n" + page.page_content

        return context.strip() if context.strip() else "No relevant data found."

    


    pdf_tool = Tool(
        name="pdf_search",
        func=search_pdf,
        description="Search the uploaded PDFs for relevant information."
    )
    tools =[]
    tools.insert(0, pdf_tool) 
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
    result = st.session_state.agent_executor.invoke({"input": user_question + " from the document"})
    answer = result["output"]

    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({"role": "assistant", "content": answer})
    #st.chat_message("assistant").write(answer)

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

