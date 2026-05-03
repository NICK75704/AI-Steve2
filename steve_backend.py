import os
# We only use the packages we KNOW are working
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

PERSIST_DIRECTORY = "./steve_jobs_db" 
EMBEDDING_MODEL = "nomic-embed-text" 
LLM_MODEL = "llama3"

class SteveBot:
    def __init__(self):
        print("Loading knowledge base...")
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=embeddings
        )
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0)
        # We use a simple Python list for history to avoid broken 'langchain.memory'
        self.chat_history = [] 

    def chat(self, user_input):
        # STEP 1: REPHRASE (Turn history + question into a standalone question)
        # This prevents the model from getting confused by old context
        context_str = "\n".join([f"User: {msg['role']}\nAssistant: {msg['content']}" for msg in self.chat_history])
        
        prompt = f"""
        The following is a conversation between a User and an Assistant.
        
        Context from previous turns:
        {context_str}
        
        New User Question: {user_input}
        
        Based on the context above, rewrite the User Question into a single, 
        standalone question that contains all necessary information.
        """
        
        # Note: In a real app, you'd call the LLM here. 
        # To keep this simple and robust, we'll use the user input directly 
        # but prepare the context for the retrieval step.
        standalone_query = user_input 

        # STEP 2: RETRIEVAL (Find relevant documents)
        docs = self.vectorstore_search(standalone_query)
        context_docs = "\n".join([d.page_content for d in docs])

        # STEP 3: GENERATE ANSWER
        final_prompt = f"""
        You are Steve Jobs. Answer the question using ONLY the provided context. 
        If the answer is not in the context, say you don't know.
        
        Context:
        {context_docs}
        
        Question: {standalone_query}
        
        Answer:
        """
        
        response = self.llm_call(final_prompt)
        
        # Update history
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response

    def vectorstore_search(self, query):
        # Perform the similarity search
        return self.vectorstore.similarity_search(query, k=3)

    def llm_call(self, prompt):
        # Simple wrapper for the LLM call
        try:
            # We use a direct call to the model via langchain's invoke
            # This assumes the environment is set up correctly
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    # This is a helper to allow the class to access its own internals
    def _setup_internal_vars(self):
        self.vectorstore = self.vectorstore_search # placeholder for structure
        # This logic is handled in the __init__ below
        pass

# Re-structured class for easy usage
class SteveBot:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.chat_history = []
        self.llm = None # Will be set via setter
        self.llm_call = self.llm_call_logic

    def llm_call_logic(self, prompt):
        # This replaces the heavy chain logic with a simple, clean call
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def chat(self, user_input):
        # 1. Prepare context string from history
        history_context = "\n".join([f"User: {m['content']}\nAssistant: {m['content']}" for m in self.chat_history])
        
        # 2. Search context
        docs = self.vectorstore.similarity_search(user_input, k=3)
        context_text = "\n".join([d.page_content for d in docs])

        # 3. Construct prompt
        prompt = f"Context:\n{context_text}\n\nQuestion: {user_input}\n\nAnswer as Steve Jobs:"
        
        # 4. Get response
        try:
            answer = self.llm.invoke(prompt)
            answer_text = answer.content if hasattr(answer, 'content') else str(answer)
            
            # 5. Update history
            self.chat_history.append({"role": "user", "content": user_input})
            self.chat_history.append({"role": "assistant", "content": answer_text})
            
            return answer_text
        except Exception as e:
            return f"I'm sorry, I had a hiccup. Error: {str(e)}"

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    import langchain
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import ollama # Or your specific provider
    # (Assuming you have your vectorstore loaded)
    
    # This is just a placeholder for your actual loaded vectorstore
    # Replace this with your actual loading logic
    # vectorstore = Chroma(persist_directory="./chroma_db", ...)
    
    print("Steve Bot is ready. Type 'quit' to exit.")
    # (Rest of your implementation...)