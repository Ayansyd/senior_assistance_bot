import sys
import logging
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
 
# --- Configuration (Match rag_setup.py) ---
PERSIST_DIRECTORY = "vector_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
NUM_RESULTS_TO_FETCH = 4 # How many results to show
 
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
 
def query_vector_store(query_text: str):
    """Loads the vector store and performs a similarity search."""
 
    logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    try:
        # Ensure you have sentence-transformers installed: pip install sentence-transformers
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logging.info("Embedding model loaded.")
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}", exc_info=True)
        return
 
    logging.info(f"Loading vector store from: {PERSIST_DIRECTORY}")
    try:
        # Ensure you have chromadb installed: pip install chromadb
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        logging.info("Vector store loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load vector store from '{PERSIST_DIRECTORY}': {e}", exc_info=True)
        logging.error("Ensure the vector_db directory exists and was created successfully.")
        return
 
    logging.info(f"\nPerforming similarity search for: '{query_text}'")
    try:
        # Perform the search
        results = vector_store.similarity_search(
            query=query_text,
            k=NUM_RESULTS_TO_FETCH # Fetch top k results
        )
 
        if not results:
            print("\nNo relevant documents found.")
            return
 
        print(f"\n--- Top {len(results)} Results ---")
        for i, doc in enumerate(results):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', None)
            source_str = f"Source: {source}"
            if page is not None:
                 source_str += f" (Page ~{page})" # Page numbers might be approximate
 
            print(f"\n{i+1}. {source_str}")
            print(f"   Content: {doc.page_content}")
            # print(f"   Metadata: {doc.metadata}") # Uncomment to see all metadata
            print("-" * 20)
 
    except Exception as e:
        logging.error(f"Error during similarity search: {e}", exc_info=True)
 
 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        search_query = " ".join(sys.argv[1:]) # Combine all arguments into query
        query_vector_store(search_query)
    else:
        print("Usage: python query_db.py <your search query>")
        # Example interactive query
        try:
             while True:
                  query = input("\nEnter your query (or Ctrl+C to exit): ")
                  if query.strip():
                       query_vector_store(query.strip())
                  else:
                       break
        except KeyboardInterrupt:
             print("\nExiting.")