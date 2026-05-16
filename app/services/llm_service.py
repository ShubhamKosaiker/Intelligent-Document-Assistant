from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from config import Config

class LLMService:
    def __init__(self,vector_store):
        self.llm = ChatGroq(
        temperature=0.6,
        model_name="llama-3.3-70b-versatile",
        groq_api_key=Config.GROQ_API_KEY
        )
        # Windowed memory (k=4) prevents unbounded prompt growth that was
        # driving P95 e2e latency from ~1s to >20s in long sessions.
        self.memory= ConversationBufferWindowMemory(
            k=4,
            memory_key="chat_history",
            return_messages=True
        )

        self.chain= ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vector_store.vector_store.as_retriever(),
            memory=self.memory
        )

    def get_response(self,query):
        try:
            response =self.chain.invoke({"question":query})
            return response['answer']
        except Exception as e:
            print(f"Error getting LLM Response : {e}")
            return "I encountered an error processing your request."