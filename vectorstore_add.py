import time
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException
import pandas as pd
import numpy as np
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain.schema.document import Document
import os
load_dotenv()

# Get the API_KEY
api_key = os.getenv('API_KEY')
print(api_key)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('allenai-specter')

# change the path according to your directory
DATA_PATH = r'C:\QpiAi'

def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()
def split_documents(documents:list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def metadata_adding(doc, stored_meta):
    document = doc
    stored_metadata = stored_meta

    for doc in document:
        # Extract the paper ID from the source filename
        source_path = doc.metadata['source']
        
        # Find the corresponding metadata entry using pandas DataFrame filtering
        matching_metadata = stored_metadata[
            stored_metadata['arxiv_id'].astype(str).str.contains(
                source_path.split('_')[-1].replace('.pdf', '')
            )
        ]

        if not matching_metadata.empty:
            # Access values from the matching metadata entry
            author = matching_metadata.iloc[0]['Author']
            publication_date = matching_metadata.iloc[0]['publication_date']
            title = matching_metadata.iloc[0]['Title']
            abstract = matching_metadata.iloc[0]['Abstract']
            
            # Update document metadata
            doc.metadata['Author'] = author
            doc.metadata['publication_date'] = publication_date
            doc.metadata['Title'] = title
            doc.metadata['Abstract'] = abstract

    return document 

def add_to_pinecone(chunks: list[Document],batch_size = 100):
    pc = Pinecone(api_key=api_key)
    index_name = 'specter'

    # Initialize Pinecone
    try:
        if index_name not in pc.list_indexes():
            pc.create_index(name=index_name, dimension=384, spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"))
            print('✅ Creating new index')
    except PineconeApiException as e:
        print(f'✅ Index "{index_name}" already exists')

    # Connect to the existing index
    index = pc.Index(index_name)

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)
    chunks_with_ids = metadata_adding(chunks_with_ids, stored_meta)

    # Fetch existing IDs
    existing_ids = set()
    ids_to_check = [chunk.metadata["id"] for chunk in chunks_with_ids]

    try:
        fetch_response = index.fetch(ids=ids_to_check)
        if fetch_response and 'vectors' in fetch_response:
            existing_ids = set(fetch_response['vectors'].keys())
    except Exception as e:
        print(f"Error fetching existing IDs: {e}")

    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Check for new chunks
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
        else:
            print(f"Duplicate found for ID: {chunk.metadata['id']} - not adding.")

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")

        # Prepare all new chunks for upsert
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        batch_texts = [str(chunk.page_content) for chunk in new_chunks]  
        embedded_texts = [model.encode(text) for text in batch_texts] 
        
        vectors_with_metadata = []
        
        for i in range(0, len(new_chunks), batch_size):
            vectors_with_metadata = []

            # Process a batch of chunks
            batch_chunks = new_chunks[i:i + batch_size]
            batch_embedded_texts = embedded_texts[i:i + batch_size]

            for idx, embedding in enumerate(batch_embedded_texts):
                authors_list = [author.strip() for author in batch_chunks[idx].metadata.get('Author', 'Unknown').split(',')]

                vectors_with_metadata.append({
                    'id': batch_chunks[idx].metadata["id"],
                    'values': embedding,
                    'metadata': {
                        'text': batch_texts[idx],  # Store the text as metadata
                        'Author': authors_list,  # Author metadata
                        'publication_date': batch_chunks[idx].metadata.get('publication_date', 'Unknown'),  # Publication date metadata
                        'Title': batch_chunks[idx].metadata.get('Title', 'Unknown'),  # Title metadata
                        'Abstract': batch_chunks[idx].metadata.get('Abstract', 'Unknown')  # Abstract metadata
                    }
                })
        
            # Upsert embeddings into Pinecone
            try:
                index.upsert(vectors=vectors_with_metadata)
                print(" All new documents added")
            except Exception as e:
                print(f"Error upserting documents: {e}")
    else:
        print(" No new documents to add")

document = load_documents()

doc = split_documents(document)
stored_meta = pd.read_csv('arxiv_metadata.csv')
add_to_pinecone(doc)
