# Local Inferencing Application with Llama

## Description
This application provides a local inferencing solution using the Llama model. It allows users to upload documents, process them, and query the processed information using a RAG (Retrieval-Augmented Generation) pipeline.

## Features
- PDF text extraction
- Text preprocessing and chunking
- Document embedding using BAAI/bge-small-en-v1.5
- Vector storage with ChromaDB
- Query processing with Llama 2 7B Chat model
- FastAPI-based REST API

## Prerequisites
- Python 3.7+
- CUDA-capable GPU (optional, for faster processing)

## Installation

1. Clone the repository:

git clone https://github.com/lokeshreddym/LlamaRAG-LocalInference.git
cd LlamaRAG-LocalInference


2. Install the required packages:

pip install -r requirements.txt

3. Download the Llama 2 7B Chat model and place it in the specified path:
llama-2-7b-chat.Q4_K_M.gguf is available in huggingface https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

## Usage

1. Start the FastAPI server:
python main.py

2. The server will start running on `http://0.0.0.0:8000`

3. Use the API endpoints to upload documents and query the system.

## API Endpoints

1. Upload Document:
- URL: `/upload_document`
- Method: POST
- Body: Form-data with key 'file' and value as the PDF file

2. Query:
- URL: `/query`
- Method: POST
- Body: JSON
  ```json
  {
    "question": "Your question here"
  }
  ```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)