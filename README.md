# RAG-based-LLM-Q-A-Chatbot-App-using-Named-Entity-Recognition-

This repository contains a Gradio application for an RAG Q&A bot with port 8760 that leverages LLaMA 3.1 and Pinecone Vector Database. The app allows users to interact with the Model and get responses based on their queries.

## DataSet

- Research Paper (pdf) : Multi-View and Multi-Scale Alignment for Contrastive Language-Image Pre-training in Mammography
- Research Paper (pdf) : DIFFUSION-BASED VISUAL FOUNDATION MODEL FOR HIGH-QUALITY DENSE PREDICTION
- Research Paper (pdf) : LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D-awareness
- Research Paper (pdf) : EGOLM: MULTI-MODAL LANGUAGE MODEL OF EGOCENTRIC MOTIONS
- Research Paper (pdf) : FlowTurbo: Towards Real-time Flow-Based Image Generation with Velocity Refiner

## Installation

## Overview of files
### Data_Ingestion.py = Scrap the arxiv website to download pdfs to the working dir and store the metadata in csv format.
![metadata](https://github.com/user-attachments/assets/e314da9e-07a0-473a-9f5e-f1d8ea588690)

### Vectorstore.py = Converted text into embeddings and stored in pinecone vector database
- Working first it will fetch the id's from vector_database compare with the ids that needs to be upserted
- if new ids are detected
![image](https://github.com/user-attachments/assets/cabe362b-16e9-4717-ba55-de17440471e9)
- else if the ids are already present in vector store then
![image](https://github.com/user-attachments/assets/9dd60749-f4f6-4f9d-a729-0db3734f0b15)

### query_chat.py 
- added functionality of Named entity Recognition
- use Case if a user wants to view research paper by the author's or publication date? Rag-based llm cannot find similarity of chunks as it is cosin
- Work Flow user query -> converted to NER (spacy) -> if Person is detected -> Do the metadata filtering with that Author
####Example

- Without NER (retrieval extracts chunks of different research paper's (Wrong Output)
- 
![before_ner](https://github.com/user-attachments/assets/e97a25c9-a1b1-484d-9ec4-344dc6a7c953)
- With NER
- 
![after NER](https://github.com/user-attachments/assets/e6eef061-123a-4479-94c4-3c65b4990701)

### ChatBot Interface (Gradio) running in port 8760 

![image](https://github.com/user-attachments/assets/b83b90aa-45a6-4c34-b117-51b65c4e2a42)

- Same prompt can be found in ml flow at port 5000 (for tracking)
- Every time user is going to query in the chatbot, mlflow will log the 4 params (query,clarification needed, highest_score(Critical for analysing different techniques) and Generated response
![2 params](https://github.com/user-attachments/assets/59e17a2b-9908-498e-ac14-aeae153b8c78)
- Parameters and response are then stored in local directory
  ![local dir](https://github.com/user-attachments/assets/96c3b50f-aecf-4b3f-8e78-ebdcbbcdcb10)

### Unit Testing using PyTest for (model evaluation)


## Installation

1. Clone the repository:

```sh
git clone https://github.com/s3dhanth/RAG-Chatbot-utilising-Name-Entity-Recognition.git

cd RAG-Chatbot-utilising-Name-Entity-Recognition
sh
Copy code
pip install -r requirements.txt

run python dataingestion.py
run python vectorstore.py

## to run chatbot
run python query_chat_with_mlflow.py

## Using Docker
- docker-compose up --build

# Gradio Application
- Go to port 7860 (local host) for gradio
- Go to port 5000 (local host) for mlflow


```


  
