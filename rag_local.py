import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------------------------------------------
#                 CONFIGURATION
# -------------------------------------------------------
load_dotenv()  # Load .env file if exists

DATA_PATH = "/home/griffyn-project/Prajwal/try/ollama/"
PDF_FILENAME = "sop.pdf"   # Change to your file name
FAISS_PATH = "faiss_index"

# -------------------------------------------------------
#              DOCUMENT LOADING & SPLITTING
# -------------------------------------------------------

def load_documents():
    """Loads documents from the specified data path."""
    pdf_path = os.path.join(DATA_PATH, PDF_FILENAME)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} page(s) from {pdf_path}")
    return documents

def split_documents(documents):
    """Splits documents into smaller overlapping text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"🧩 Split into {len(all_splits)} chunks")
    return all_splits

# -------------------------------------------------------
#                EMBEDDING FUNCTION
# -------------------------------------------------------

def get_embedding_function(model_name="nomic-embed-text"):
    """Initializes the Ollama embedding model."""
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"🧠 Using embedding model: {model_name}")
    return embeddings

# -------------------------------------------------------
#                FAISS VECTOR STORE
# -------------------------------------------------------

def index_documents(chunks, embedding_function, persist_directory=FAISS_PATH):
    """Indexes document chunks using FAISS (no SQLite required)."""
    print(f"⚙️  Indexing {len(chunks)} chunks using FAISS...")

    # Create a FAISS index
    vectorstore = FAISS.from_documents(chunks, embedding_function)

    # Save locally (FAISS supports local save/load)
    vectorstore.save_local(persist_directory)
    print(f"✅ Indexing complete. FAISS index saved to: {persist_directory}")
    return vectorstore

def load_faiss_index(embedding_function, persist_directory=FAISS_PATH):
    """Loads an existing FAISS index if available."""
    if os.path.exists(persist_directory):
        print(f"📂 Loading existing FAISS index from {persist_directory}")
        return FAISS.load_local(persist_directory, embedding_function, allow_dangerous_deserialization=True)
    else:
        print("⚠️ No existing FAISS index found. You may need to index documents first.")
        return None

# -------------------------------------------------------
#                RAG CHAIN CREATION
# -------------------------------------------------------

def create_rag_chain(vector_store, llm_model_name="qwen3:8b", context_window=8192):
    """Creates the RAG pipeline using FAISS retriever + Ollama LLM."""
    llm = ChatOllama(
        model=llm_model_name,
        temperature=0,
        num_ctx=context_window
    )
    print(f"🤖 Initialized ChatOllama with model: {llm_model_name}")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    print("🔍 Retriever ready.")

    template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    print("🧱 Prompt template created.")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("🔗 RAG chain created successfully.")
    return rag_chain

# -------------------------------------------------------
#                 QUERY HANDLER
# -------------------------------------------------------

def query_rag(chain, question):
    """Runs a query through the RAG chain and prints the answer."""
    print("\n🟢 Querying RAG chain...")
    print(f"❓ Question: {question}")
    response = chain.invoke(question)
    print("\n💬 Response:")
    print(response)

# -------------------------------------------------------
#                     MAIN
# -------------------------------------------------------

if __name__ == "__main__":
    # 1. Load Documents
    docs = load_documents()

    # 2. Split Documents
    chunks = split_documents(docs)

    # 3. Initialize Embedding Function
    embedding_function = get_embedding_function()

    # 4. Build or Load FAISS Index
    if not os.path.exists(FAISS_PATH):
        vector_store = index_documents(chunks, embedding_function)
    else:
        vector_store = load_faiss_index(embedding_function)

    # 5. Create RAG Chain
    rag_chain = create_rag_chain(vector_store, llm_model_name="qwen3:8b")

    # 6. Ask Questions
    query_rag(rag_chain, "Explain the AI pipeline in the document")
    query_rag(rag_chain, "Summarize the introduction section.")

