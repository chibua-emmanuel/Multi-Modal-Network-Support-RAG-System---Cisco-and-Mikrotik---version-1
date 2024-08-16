import streamlit as st
import os
import pytesseract
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
import ollama

class RAGSystem:
    def __init__(self, openai_api_key: str, model: str):
        self.openai_api_key = openai_api_key
        self.model = model
        self.vector_store = None
        self.chat_history = []  # Start with an empty chat history

    def pdf_to_langchain_document_parser(self, files):
        """Parse and split PDF documents into pages."""
        pages = []
        for file in files:
            try:
                with open(file.name, "wb") as f:
                    f.write(file.getbuffer())
                loader = PyPDFLoader(file.name)
                pages.extend(loader.load_and_split())
            except Exception as e:
                st.error(f"Error parsing PDF {file.name}: {e}")
        st.success(f"Loaded {len(pages)} pages from the PDF documents.")
        return pages

    def image_to_text(self, file):
        """Extract text from an image using OCR."""
        try:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
            st.success("Text extracted from image successfully!")
            return text
        except Exception as e:
            st.error(f"Error extracting text from image: {e}")
            return ""

    def get_text_chunks(self, text: str):
        """Split text into manageable chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return text_splitter.split_text(text)

    def get_vector_store(self, text_chunks: list):
        """Create a vector store using OpenAI embeddings."""
        try:
            embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
            st.success("Vector store created successfully!")
            return vector_store
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return None

    def get_conversational_chain(self):
        """Set up the question-answering conversational chain."""
        if self.model == "gpt-4o":
            prompt_template = """
            You are a knowledgeable subject matter expert in networks and telecommunication specializing in Cisco and Mikrotik devices. Provide helpful and relevant answers based on the given context.
            Context:\n {context}?\n
            Question: \n{question}\n
            Answer:
            """
            model = ChatOpenAI(model="gpt-4o", openai_api_key=self.openai_api_key)
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            return load_qa_chain(model, chain_type="stuff", prompt=prompt)
        elif self.model == "llama3.1":
            def llama_chat(context, question):
                response = ollama.chat(model='llama3.1', messages=[
                    {
                        'role': 'system',
                        'content': 'You are a knowledgeable subject matter expert in networks and telecommunication specializing in Cisco and Mikrotik devices. Provide helpful and relevant answers based on the given context.'
                    },
                    {
                        'role': 'user',
                        'content': f"Context: {context}\nQuestion: {question}\nAnswer:",
                    },
                ])
                return response['message']['content']
            return llama_chat

    def user_input(self, user_question: str):
        """Process user input and return a response."""
        if self.vector_store is None:
            st.error("RAG system has not been initialized. Please run initialize_rag first.")
            return "Initialization required."

        self.chat_history.append({'role': 'user', 'content': user_question})
        
        try:
            docs = self.vector_store.similarity_search(user_question)
            chain = self.get_conversational_chain()
            if self.model == "gpt-4o":
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
                self.chat_history.append({'role': 'assistant', 'content': response['output_text']})
                return response['output_text']
            elif self.model == "llama3.1":
                context = " ".join([doc.page_content for doc in docs])
                response = chain(context, user_question)
                self.chat_history.append({'role': 'assistant', 'content': response})
                return response
        except Exception as e:
            st.error(f"Error during question processing: {e}")
            return "An error occurred while processing your request."

    def initialize_rag(self, files):
        """Initialize the RAG system with selected PDF documents."""
        pdf_docs = self.pdf_to_langchain_document_parser(files)
        if pdf_docs:
            self.vector_store = self.get_vector_store(pdf_docs)
            return self.vector_store is not None  # Return True if initialization is successful
        else:
            st.error("Failed to initialize RAG system due to PDF loading issues.")
            return False

# Streamlit interface
def main():
    st.title("Network Support RAG System - Cisco & Mikrotik")
    st.write("NETA AI")

    openai_api_key = "..."

    # Model selection
    model_option = st.selectbox("Select Model", ["gpt-4o", "llama3.1"])

    # Initialize RAG System with session state
    if "rag_system" not in st.session_state:
        st.session_state["rag_system"] = RAGSystem(openai_api_key=openai_api_key, model=model_option)
        st.session_state["initialized"] = False

    # Text entry for user input at the top of the page
    user_question = st.text_input("Enter your question here")

    # File uploader for PDF documents
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # File uploader for images
    uploaded_image = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    # Display checkboxes for selecting PDF files
    if uploaded_files:
        selected_files = []
        st.write("Select PDF files to use for RAG initialization:")
        for file in uploaded_files:
            if st.checkbox(file.name):
                selected_files.append(file)

    # Initialize RAG System button
    if st.button("Initialize RAG System"):
        if selected_files:
            try:
                st.session_state["initialized"] = st.session_state["rag_system"].initialize_rag(selected_files)
                if st.session_state["initialized"]:
                    st.success("RAG system initialized successfully!")
                else:
                    st.error("RAG system initialization failed.")
            except Exception as e:
                st.error(f"Error initializing RAG system: {e}")
        else:
            st.warning("Please select PDF files to initialize the RAG system.")

    # Process Image button
    if st.button("Process Image"):
        if uploaded_image is not None:
            text_from_image = st.session_state["rag_system"].image_to_text(uploaded_image)
            st.write("**Extracted Text from Image:**")
            st.write(text_from_image)
        else:
            st.warning("Please upload an image file to process.")

    # Ask Question button
    if st.button("Ask Question", key="ask_button"):
        if not st.session_state["initialized"]:
            st.warning("Please initialize the RAG system first.")
        elif not user_question:
            st.warning("Please enter a question.")
        else:
            response = st.session_state["rag_system"].user_input(user_question)
            if response != "Initialization required.":
                st.write("**Answer:**")
                st.write(response)
            else:
                st.warning("Please ensure the RAG system is initialized before asking questions.")

    # Ask Question from Image button
    if st.button("Ask Question from Image"):
        if uploaded_image is not None:
            text_from_image = st.session_state["rag_system"].image_to_text(uploaded_image)
            if text_from_image:
                response = st.session_state["rag_system"].user_input(text_from_image)
                st.write("**Answer:**")
                st.write(response)
            else:
                st.warning("No text extracted from the image.")
        else:
            st.warning("Please upload an image file to process.")

     # Display chat history
    st.write("### Chat History")
    if st.session_state["rag_system"].chat_history:
        for entry in st.session_state["rag_system"].chat_history:
            if entry['role'] == 'user':
                st.write(f"**User:** {entry['content']}")
            else:
                st.write(f"**Assistant:** {entry['content']}")

if __name__ == "__main__":
    main()