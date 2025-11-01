import streamlit as st
import os
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


from dotenv import load_dotenv
load_dotenv()

## Load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']  


if "vectors" not in st.session_state:
    urls = [
        "https://huggingface.co/docs/transformers/index",
        "https://huggingface.co/docs/datasets/index",
        "https://huggingface.co/docs/tokenizers/index",
        "https://huggingface.co/docs/hub/index",
        "https://huggingface.co/docs/accelerate/index",
        "https://huggingface.co/docs/optimum/index"
    ]

    st.session_state.docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        st.session_state.docs.extend(loader.load())

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

llm = ChatGroq(groq_api_key = groq_api_key, 
               model_name = "llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
"""
Answer the question based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
</context>
Question:{input}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# ----------------- Header Section -----------------
st.markdown("""
    <style>
    # .main-title {
    #     font-size: 3rem;
    #     font-weight: 700;
    #     text-align: center;
    #     color: #4B8BBE;
    #     margin-top: 20px;
    #     margin-bottom: 5px;
    # }
    .sub-title {
        text-align: center;
        font-size: 1.2rem;            
        color: #888;
        margin-bottom: 30px;
    }
    .response-box {
        background-color: #1e1e2f;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-top: 15px;
    }
    .doc-box {
        background-color: #0f4c75;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #4B8BBE;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE; font-size: 3rem;'>
        ChatGroq RAG Assistant
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="sub-title">An AI chatbot that answers questions from Hugging Face documentation using RAG & FAISS</p>', unsafe_allow_html=True)

# ----------------- Input Section -----------------
st.markdown("### 🧠 Ask me anything about Hugging Face documentation:")
prompt = st.text_input("Type your question here...", placeholder="e.g. What does the pipeline() function in Transformers do?")

# ----------------- Chat Response Section -----------------
if prompt:
    with st.spinner("🔍 Generating answer..."):
        start = time.process_time()
        response = retrieval_chain.invoke({"input": prompt})
        response_time = round(time.process_time() - start, 2)

    # Display answer
    st.markdown("### 🪄 Answer:")
    st.markdown(f'<div class="response-box">{response["answer"]}</div>', unsafe_allow_html=True)
    st.markdown(f"⏱️ *Response generated in {response_time} seconds*")

    # Document similarity search section
    with st.expander("📚 View Retrieved Context Documents"):
        for i, doc in enumerate(response["context"]):
            st.markdown(f'<div class="doc-box">{doc.page_content}</div>', unsafe_allow_html=True)

# ----------------- Footer -----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#777; font-size:0.9rem;'>"
    "Built with ❤️ using Streamlit, LangChain, Groq API & FAISS"
    "</p>",
    unsafe_allow_html=True
)
