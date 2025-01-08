from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

class RAGChatbot:
    def __init__(self, data_path):
        self.data_path = data_path
        self.documents = None
        self.vector_store = None
        self.qa_chain = None
        self.setup_pipeline()

    def load_data(self):
        loader = CSVLoader(self.data_path)
        self.documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.documents = text_splitter.split_documents(self.documents)

    def create_vector_store(self):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(self.documents, embeddings)

    def setup_pipeline(self):
        self.load_data()
        self.create_vector_store()
        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.1})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=self.vector_store.as_retriever(), memory=memory)

    def ask(self, question):
        try:
            result = self.qa_chain.invoke({"question": question})
            return result
        except Exception as e:
            return {"answer": f"Error: {str(e)}"}

if __name__ == "__main__":
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("Error: HUGGINGFACEHUB_API_TOKEN environment variable not set")
        exit(1)

    DATA_PATH = "../Langbot/data.csv"
    chatbot = RAGChatbot(DATA_PATH)
    
    while True:
        question = input("Ask a question (or type 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        response = chatbot.ask(question)
        print("\nAnswer:", response["answer"])


# export OPENAI_API_KEY='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'




# Hugging face access token = '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'