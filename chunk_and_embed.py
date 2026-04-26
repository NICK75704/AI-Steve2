import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_FOLDER = "./Steve_Knowledge" 
PERSIST_DIRECTORY = "./steve_jobs_db" 
EMBEDDING_MODEL = "nomic-embed-text" 

def create_embeddings():
    print(f"Loading documents from {DOCS_FOLDER}...")
    if not os.path.exists(DOCS_FOLDER):
        print(f"Error: Folder {DOCS_FOLDER} not found. Please create it and add .txt files.")
        return

    loader = DirectoryLoader(DOCS_FOLDER, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} text chunks.")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print("Embedding and saving to disk (this may take a minute)...")
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    
    vector_db.persist()
    print(f"Success! Your knowledge base is saved at: {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    create_embeddings()