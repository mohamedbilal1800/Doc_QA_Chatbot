import os
import tempfile
import chromadb
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
CHROMA_DB_PATH = "chroma_db"


# Caching the embedding model for faster response
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def get_vector_store():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return Chroma(
        collection_name="qa_bot",
        embedding_function=get_embedding_model(),
        client=chroma_client
    )


@st.cache_resource
def get_chat_model():
    # groq_api_key = os.getenv("GROQ_API_KEY")
    groq_api_key = st.secrets["GROQ_API_KEY"]
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Please set it in the .env file.")
    return ChatGroq(api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")


# prompt template for RAG
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an AI assistant answering questions based on the provided documents and your own knowledge.\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Provide a detailed answer using the relevant information."
    )
)


@st.cache_data
def retrieve_context(query, k=3):
    """Retrieve top-k most relevant document chunks from ChromaDB."""
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    return docs


@st.cache_data
def generate_rag_response(query):
    """Generate a response using retrieved document context and ChatGroq."""
    docs = retrieve_context(query)

    if not docs:
        return "No relevant documents found. Please try a different question.", []

    context = "\n".join([doc.page_content for doc in docs])
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]

    # Format the prompt
    prompt = prompt_template.format(context=context, question=query)

    # Call ChatGroq model
    chat_model = get_chat_model()
    response = chat_model.invoke([HumanMessage(content=prompt)])
    return response.content, sources


# Function to process and store uploaded documents
def load_and_store_documents(file_path, file_name):
    """Loads documents, splits them into chunks, and stores embeddings in ChromaDB."""
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a .pdf or .txt file.")

    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Store embeddings in ChromaDB
    vector_store = get_vector_store()
    vector_store.add_texts(
        texts=[chunk.page_content for chunk in chunks],
        metadatas=[{"source": file_path} for chunk in chunks]
    )

    return len(chunks)


# Streamlit UI for chatbot
def main():
    st.set_page_config(page_title="📄 ChatGroq DeepseekAI Chatbot", layout="wide")
    st.sidebar.subheader("📄 ChatGroq-Deepseek Chatbot")
    st.sidebar.write("Upload a document and ask questions about its content.")

    # File Upload Section
    uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            file_path = temp_file.name

        # calling function to process and store the embeddings in chroma_db
        num_chunks = load_and_store_documents(file_path, uploaded_file.name)
        st.success(f"✅ Processed and stored {num_chunks} document chunks in ChromaDB.")

    # Initializing chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # To display past chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat User Input
    user_query = st.chat_input("Ask a question about the uploaded documents...")

    if user_query:
        # Displaying user query
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Generate response
        response, sources = generate_rag_response(user_query)

        # Display AI response
        with st.chat_message("assistant"):
            st.write(response)

        # Save response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
