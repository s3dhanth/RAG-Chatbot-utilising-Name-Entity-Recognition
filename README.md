# RAG-based-LLM-Q-A-Chatbot-App-using-Named-Entity-Recognition-

This repository contains a Gradio application for an RAG Q&A bot with port 8760 that leverages LLaMA 3.1 and Pinecone Vector Database. The app allows users to interact with the Model and get responses based on their queries.

## DataSet

- Research Paper (pdf) : Multi-View and Multi-Scale Alignment for Contrastive Language-Image Pre-training in Mammography
- Research Paper (pdf) : DIFFUSION-BASED VISUAL FOUNDATION MODEL FOR HIGH-QUALITY DENSE PREDICTION
- Research Paper (pdf) : LLaVA-3D: A Simple yet Effective Pathway to Empowering LMMs with 3D-awareness
- Research Paper (pdf) : EGOLM: MULTI-MODAL LANGUAGE MODEL OF EGOCENTRIC MOTIONS
- Research Paper (pdf) : FlowTurbo: Towards Real-time Flow-Based Image Generation with Velocity Refiner

## Overview of files
### Data_Ingestion.py = Scrap the arxiv website to download pdfs to the working dir and store the metadata in csv format.
![metadata](https://github.com/user-attachments/assets/e314da9e-07a0-473a-9f5e-f1d8ea588690)

### Vectorstore.py = Converted text into embeddings and stored in pinecone vector database
- Working first it will fetch the id's from vector_database compare with the ids that needs to be upserted
- if new ids are detected
![image](https://github.com/user-attachments/assets/cabe362b-16e9-4717-ba55-de17440471e9)
- else if the ids are already present in vector store then
![image](https://github.com/user-attachments/assets/9dd60749-f4f6-4f9d-a729-0db3734f0b15)

### quey_chat.py 
- added functionality of Named entity Recognition
- use Case if a user wants to view research paper by the author's or publication date? Rag-based llm cannot find similarity of chunks as it is cosin
- Work Flow user query -> converted to NER (spacy) -> if Person is detected -> Do the metadata filtering with that Author
####Example
- Without NER (retrieval extracts chunks of different research paper's (Wrong Output)
![before_ner](https://github.com/user-attachments/assets/e97a25c9-a1b1-484d-9ec4-344dc6a7c953)
- With NER
![after NER](https://github.com/user-attachments/assets/e6eef061-123a-4479-94c4-3c65b4990701)

### ChatBot Interface 
  
### Sentence Transformer (MiniLM) Model Embedding after dimension reduction
![image](https://github.com/user-attachments/assets/fb82b515-3bfc-47cb-96fb-21694d65dde1)

### Query Expansion (using NLTK and Wordnet)
- removed Stop words and punctuations
- Added synonyms using wordnet

### Hybrid Search with sparse-dense vector

![image](https://github.com/user-attachments/assets/4ad8c914-746a-4072-b1c4-e93a57066a68)

## Installation

1. Clone the repository:

```sh
git clone https://github.com/s3dhanth/Raptor-based-LLM-Q-A-Chatbot-App-using-hybrid-search-.git

cd Raptor-based-LLM-Q-A-Chatbot-App-using-hybrid-search-
sh
Copy code
pip install -r requirements.txt

run python main.py (llama3.1)
          or
run python Sentence_transformer.py (Transformer)

-to Ask custom question use (rag.invoke("question")) 

# Gradio Application
- # run python app.py (llama3.1 embeddings)
- # run python Sentence_transformers.py (MiniLM model embeddings)

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
```
## Model (Sentence Transformer MiniLM) 
- In the text box Enter our query and submit 
![minilm](https://github.com/user-attachments/assets/cd40599d-2576-43c2-b099-99b760715919)

## Query vs Expanded Query (Unit Testing)
- **Query**
![image](https://github.com/user-attachments/assets/6cef66d0-f892-4462-b091-03ae6f4b03f4)
Model Failed to decode the question
- **Expanded Query**
Original Question(Who is the owner of the dining venue?)
![image](https://github.com/user-attachments/assets/42bc3fd5-af7c-4f67-8c8a-1ad0b074c63a)


## Notebooks overview
- Raptor_sbert_2dim -   Reduction to 10 Dimensions
- Raptor_sbert_2dim -   Reduction to 2 Dimensions
- MiniLMapp - Sentence Transformer model with gradio (jupyter)
- Sentence_Transformer = For Model testing (use it directly)
- Main.py = For LLama embeddings Model testing (use it directly)
  
