from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from rich.console import Console
from rich.markdown import Markdown
import os
import time

class RAGChatApplication:
    def __init__(self, data_path, hf_token):
        """Initialize the RAG Chat Application with data path and HuggingFace token."""
        self.console = Console()
        self.data_path = data_path
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        self.documents = None
        self.vector_store = None
        self.qa_chain = None
        self.setup_pipeline()

    def display_welcome(self):
        """Display welcome message and instructions."""
        welcome_text = """
# RAG Chat Application
Welcome to the Retrieval-Augmented Generation Chat Application!

## Available Commands:
- Type your question normally to get an answer
- Type 'history' to see conversation history
- Type 'clear' to clear conversation history
- Type 'quit' to exit

## Dataset Information:
This chat is knowledgeable about programming languages including:
- Creation dates
- Programming paradigms
- Typing systems
        """
        self.console.print(Markdown(welcome_text))

    def load_data(self):
        """Load and process the dataset."""
        self.console.print("[bold blue]Loading data...[/bold blue]")
        loader = CSVLoader(self.data_path)
        self.documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.documents = text_splitter.split_documents(self.documents)

    def create_vector_store(self):
        """Create vector store for document retrieval."""
        self.console.print("[bold blue]Creating vector store...[/bold blue]")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = FAISS.from_documents(self.documents, embeddings)

    def setup_pipeline(self):
        """Set up the RAG pipeline."""
        try:
            self.load_data()
            self.create_vector_store()
            
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-base",
                model_kwargs={"temperature": 0.1}
            )
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vector_store.as_retriever(),
                memory=self.memory
            )
            self.console.print("[bold green]Setup complete![/bold green]")
        except Exception as e:
            self.console.print(f"[bold red]Error during setup: {str(e)}[/bold red]")
            raise

    def display_chat_history(self):
        """Display the conversation history."""
        history = self.memory.chat_memory.messages
        if not history:
            self.console.print("[yellow]No chat history available.[/yellow]")
            return
        
        self.console.print("\n[bold]Chat History:[/bold]")
        for msg in history:
            if msg.type == "human":
                self.console.print(f"[bold blue]You:[/bold blue] {msg.content}")
            else:
                self.console.print(f"[bold green]Assistant:[/bold green] {msg.content}")

    def clear_chat_history(self):
        """Clear the conversation history."""
        self.memory.clear()
        self.console.print("[yellow]Chat history cleared.[/yellow]")

    def process_command(self, command):
        """Process special commands."""
        if command == "history":
            self.display_chat_history()
            return True
        elif command == "clear":
            self.clear_chat_history()
            return True
        return False

    def ask(self, question):
        """Process a question and return the answer."""
        try:
            self.console.print("[bold blue]Processing...[/bold blue]")
            result = self.qa_chain.invoke({"question": question})
            return result["answer"]
        except Exception as e:
            return f"Error: {str(e)}"

    def run(self):
        """Run the chat application."""
        self.display_welcome()
        
        while True:
            try:
                question = input("\nYou: ").strip()
                
                if question.lower() == 'quit':
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                    
                if self.process_command(question.lower()):
                    continue
                
                answer = self.ask(question)
                self.console.print("\n[bold green]Assistant:[/bold green]", answer)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Exiting...[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[bold red]Error: {str(e)}[/bold red]")

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "data.csv"
    
    # Get HuggingFace token
    hf_token = input("Please enter your HuggingFace token: ")
    
    try:
        # Initialize and run the chat application
        chat_app = RAGChatApplication(DATA_PATH, hf_token)
        chat_app.run()
    except Exception as e:
        Console().print(f"[bold red]Fatal error: {str(e)}[/bold red]")


        
# export OPENAI_API_KEY='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'




# Hugging face access token = "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"