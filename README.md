# Multi-Modal Network Support RAG System - Cisco and Mikrotik - version 1

## Project Overview

Welcome to the Multi-Modal Network Support RAG System project! This project leverages advanced AI models to provide expert support in networks and telecommunication, specializing in Cisco and Mikrotik devices. Users can interact with the system through text, images, and PDF documents to get relevant and helpful answers to their queries.

## Group Members

- **Gunjit Arora**
  - NU ID: 002679282
  - LinkedIn: [linkedin.com/in/gunjit-arora](https://linkedin.com/in/gunjit-arora)
- **Emmanuel Chibua**
  - NU ID: 002799484
  - LinkedIn: [linkedin.com/in/emmanuel-chibua-186ba887](https://linkedin.com/in/emmanuel-chibua-186ba887)
- **Abdulafeez Abobarin**
  - NU ID: 002922336
  - LinkedIn: [linkedin.com/in/afeez-abobarin](https://linkedin.com/in/afeez-abobarin)

## Video Demonstration

You can view the video demonstration of our project [here](https://www.youtube.com/watch?v=dQw4w9WgXcQ) (replace with your actual link).

## Project Description

The Multi-Modal Network Support RAG System is designed to assist users with network-related queries by leveraging the power of AI models like GPT-4o and LLaMA 3.1. Users can upload PDF documents, images, and ask questions directly through text input. The system processes these inputs and provides accurate and relevant answers based on the context.

### Key Features

- **Model Selection**: Users can choose between GPT-4o and LLaMA 3.1 for generating responses.
- **PDF Document Support**: Upload multiple PDF documents and select specific ones for initializing the RAG system.
- **Image Processing**: Extract text from images using OCR and process it as a query.
- **Text Input**: Directly ask questions through text input and get responses.
- **Chat History**: View the history of interactions with the system.

## Setup Instructions

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chibua-emmanuel/Multi-Modal-Network-Support-RAG-System---Cisco-and-Mikrotik---version-1.git
   cd Multi-Modal-Network-Support-RAG-System---Cisco-and-Mikrotik---version-1
   ```

2. **Install Dependencies**:
   ```bash
   pip install streamlit pytesseract Pillow langchain_community langchain langchain_openai ollama
   ```

3. **Install Tesseract OCR**:
   - **macOS**:
     ```bash
     brew install tesseract
     ```
   - **Ubuntu**:
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - **Windows**:
     Download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and follow the installation instructions.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

5. **Access the Application**:
   Open your web browser and go to `http://localhost:8501` to access the application.

## Contact Us

Feel free to reach out to us if you have any questions or need further assistance:

- **Gunjit Arora**
  - GitHub: [github.com/gunjit-arora](https://github.com/gunjit-arora)
  - LinkedIn: [linkedin.com/in/gunjit-arora](https://linkedin.com/in/gunjit-arora)

- **Emmanuel Chibua**
  - GitHub: [github.com/emmanuel-chibua](https://github.com/emmanuel-chibua)
  - LinkedIn: [linkedin.com/in/emmanuel-chibua-186ba887](https://linkedin.com/in/emmanuel-chibua-186ba887)

- **Abdulafeez Abobarin**
  - GitHub: [github.com/afeez-abobarin](https://github.com/afeez-abobarin)
  - LinkedIn: [linkedin.com/in/afeez-abobarin](https://linkedin.com/in/afeez-abobarin)

