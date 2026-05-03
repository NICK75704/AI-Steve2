from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import RetrievalQA

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_db = Chroma(persist_directory="./steve_jobs_db", embedding_function=embeddings)
llm = Ollama(model="AI-Steve:latest")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_db.as_retriever(),
    chain_type_kwargs={"prompt": "You are Steve Jobs. Use the following context to answer... [Insert full prompt from previous response]"}
)

print(qa_chain.run("What is your take on the design of the original Macintosh?"))