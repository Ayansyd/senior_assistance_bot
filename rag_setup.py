# rag_setup.py
import os
import logging
import shutil # For cleaning up old DB if needed

# Langchain components
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
# MODIFIED: Added UnstructuredWordDocumentLoader and specific imports
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader # For .docx
)
# END MODIFICATION
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging if not already configured by main app
logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Configuration ---
# Use environment variables or direct paths
KNOWLEDGE_BASE_DIR = os.getenv("KB_DIR", "knowledge_base")
PERSIST_DIRECTORY = os.getenv("VECTORDB_DIR", "vector_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Common, good quality sentence transformer
CHUNK_SIZE = 500 # How many characters per text chunk
CHUNK_OVERLAP = 50 # How much overlap between chunks

# Ensure knowledge base directory exists
if not os.path.exists(KNOWLEDGE_BASE_DIR):
    logging.warning(f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found. Creating it. Please add documents.")
    os.makedirs(KNOWLEDGE_BASE_DIR)

# --- MODIFIED: load_documents function for robustness and .docx ---
def load_documents(directory: str) -> list:
    """Loads documents from various file types in a directory."""
    logging.info(f"Loading documents from: {directory}")
    all_docs = []
    loaded_files_count = {'txt': 0, 'pdf': 0, 'docx': 0} # Track counts

    # Load TXT files
    try:
        txt_loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}, # Specify encoding
            show_progress=True,
            use_multithreading=True,
            silent_errors=True # Attempt to skip files that fail
        )
        txt_docs = txt_loader.load()
        if txt_docs:
            all_docs.extend(txt_docs)
            loaded_files_count['txt'] = len(txt_docs)
    except Exception as e:
        logging.warning(f"Could not load TXT files (or loader failed): {e}")

    # Load PDF files
    try:
        pdf_loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True,
            silent_errors=True
        )
        pdf_docs = pdf_loader.load()
        if pdf_docs:
            all_docs.extend(pdf_docs)
            loaded_files_count['pdf'] = len(pdf_docs)
    except ImportError:
        logging.warning("PyPDFLoader not available. Install 'pypdf' (`pip install pypdf`) to load PDF files.")
    except Exception as e:
        logging.warning(f"Could not load PDF files: {e}")

    # Load DOCX files
    try:
        docx_loader = DirectoryLoader(
            directory,
            glob="**/*.docx",
            loader_cls=UnstructuredWordDocumentLoader,
            show_progress=True,
            use_multithreading=True, # Unstructured might not support this well, monitor performance
            silent_errors=True
        )
        docx_docs = docx_loader.load()
        if docx_docs:
            all_docs.extend(docx_docs)
            loaded_files_count['docx'] = len(docx_docs)
    except ImportError:
         logging.warning("UnstructuredWordDocumentLoader not available. Install 'unstructured' and 'python-docx' (`pip install unstructured python-docx`) to load DOCX files.")
    except Exception as e:
        logging.warning(f"Could not load DOCX files: {e}")


    logging.info(f"Loaded {len(all_docs)} documents in total ({loaded_files_count}).")
    if not all_docs:
        # Changed from error to warning, as maybe only an existing DB is needed
        logging.warning("No documents successfully loaded from the knowledge base directory.")
    return all_docs
# --- End Modification ---


def split_documents(documents: list) -> list:
    """Splits documents into smaller chunks."""
    if not documents:
        return []
    logging.info(f"Splitting {len(documents)} documents into chunks (Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    try:
        split_docs = text_splitter.split_documents(documents)
        logging.info(f"Created {len(split_docs)} text chunks.")
        return split_docs
    except Exception as e:
        logging.error(f"Error during document splitting: {e}", exc_info=True)
        return []


def get_embedding_model():
    """Initializes the sentence transformer embedding model."""
    logging.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    # Using SentenceTransformerEmbeddings directly from langchain_community
    # device='cpu' can be specified if needed, otherwise auto-detects (incl. CUDA if available)
    try:
        # Consider adding cache_folder argument if needed: cache_folder='./embedding_cache'
        embeddings = SentenceTransformerEmbeddings(
             model_name=EMBEDDING_MODEL_NAME,
             model_kwargs={'trust_remote_code': True} # Add if facing trust issues with model loading
             )
        logging.info("Embedding model loaded successfully.")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}", exc_info=True)
        logging.error("Ensure 'sentence-transformers' library is installed and model name is correct.")
        raise # Re-raise the exception to halt initialization


def setup_vector_store(documents: list, embeddings, force_recreate: bool = False):
    """Creates or loads the Chroma vector store."""
    persist_directory = PERSIST_DIRECTORY

    if force_recreate and os.path.exists(persist_directory):
        logging.warning(f"Force recreate enabled. Removing existing vector store at: {persist_directory}")
        try:
            shutil.rmtree(persist_directory) # Remove old directory
        except OSError as e:
            logging.error(f"Error removing existing vector store directory: {e}", exc_info=True)
            # Decide whether to continue or raise error


    # Attempt to load existing store first if not forcing recreate
    if not force_recreate and os.path.exists(persist_directory):
        try:
            logging.info(f"Loading existing vector store from: {persist_directory}")
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            logging.info("Successfully loaded existing vector store.")
            # If documents were also loaded, maybe offer an update mechanism later?
            # For now, just load existing if not forcing recreate.
            return vector_store
        except Exception as e:
            logging.warning(f"Failed to load existing vector store (will attempt to recreate): {e}", exc_info=True)
            # Force recreation if loading failed but directory existed
            if os.path.exists(persist_directory):
                 logging.warning(f"Removing corrupted (?) vector store at: {persist_directory}")
                 try:
                      shutil.rmtree(persist_directory)
                 except OSError as e_rm:
                      logging.error(f"Error removing corrupted vector store directory: {e_rm}", exc_info=True)
                      # If removal fails, we probably can't proceed
                      return None


    # If store doesn't exist or we are forcing recreate, create it from documents
    logging.info("Attempting to create new vector store...")
    if not documents:
        logging.error("No documents provided or loaded, cannot create a new vector store.")
        return None

    doc_chunks = split_documents(documents)
    if not doc_chunks:
         logging.error("Document splitting resulted in zero chunks. Cannot create vector store.")
         return None

    logging.info(f"Creating new vector store at: {persist_directory} with {len(doc_chunks)} chunks.")
    try:
        vector_store = Chroma.from_documents(
            documents=doc_chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logging.info("Vector store created and persisted.")
        return vector_store
    except Exception as e:
         logging.error(f"Failed to create Chroma vector store: {e}", exc_info=True)
         return None


def get_retriever(vector_store, k: int = 3):
    """Gets a retriever object from the vector store."""
    if vector_store is None:
        logging.error("Cannot create retriever: Vector store is None.")
        return None
    logging.info(f"Creating retriever to fetch top {k} relevant documents.")
    try:
        # Configure retriever for similarity search
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        return retriever
    except Exception as e:
        logging.error(f"Failed to create retriever from vector store: {e}", exc_info=True)
        return None


# --- Main Setup Function ---
def initialize_rag(force_recreate_db: bool = False):
    """Loads documents, sets up vector store, and returns a retriever."""
    logging.info("--- Initializing RAG Components ---")
    # Load documents regardless of whether DB exists, as setup_vector_store decides creation logic
    documents = load_documents(KNOWLEDGE_BASE_DIR)

    # Check if we can proceed (either loaded docs or existing DB)
    if not documents and not os.path.exists(PERSIST_DIRECTORY) and not force_recreate_db:
        logging.error("No documents found in knowledge base and no existing vector database found. RAG cannot initialize.")
        return None

    try:
        embedding_model = get_embedding_model()
        # Pass documents - setup_vector_store will decide if they are needed
        vector_store = setup_vector_store(documents, embedding_model, force_recreate=force_recreate_db)
        retriever = get_retriever(vector_store) # get_retriever handles None vector_store
        if retriever:
             logging.info("--- RAG Initialization Complete ---")
        else:
             logging.error("--- RAG Initialization Failed (Retriever could not be created) ---")
        return retriever # Return retriever (or None if failed)
    except Exception as e:
         # Catch errors during embedding model loading specifically
         logging.error(f"Critical error during RAG initialization (likely embedding model): {e}", exc_info=True)
         return None # Return None on critical failure