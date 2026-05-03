import os
# New updated imports
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

PERSIST_DIRECTORY = "./steve_jobs_db" 
EMBEDDING_MODEL = "nomic-embed-text" 
LLM_MODEL = "AI-Steve2"

def start_chat():
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"Error: Database not found at {PERSIST_DIRECTORY}. Run your creation script first.")
        return

    print("Loading knowledge base...")
    
    # Updated classes
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings
    )

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # Removed 'return_intermediate_steps' to fix the Pydantic ValidationError
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

    print("\n" + "="*50)
    print("Steve Jobs Knowledge Assistant is Ready!")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Assistant: Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = chain.invoke({"question": user_input})
            print(f"\nAssistant: {response['answer']}\n")
            print("-" * 30)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Make sure Ollama is running locally.\n")

if __name__ == "__main__":
    start_chat()