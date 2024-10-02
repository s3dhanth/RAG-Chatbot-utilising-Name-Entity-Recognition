import gradio as gr
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from fuzzywuzzy import fuzz, process
from pinecone import Pinecone
import spacy


nlp = spacy.load("en_core_web_lg")
stored_meta = pd.read_csv('arxiv_metadata.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')
api_key = 'd7204d21-cb62-4544-b49c-9169b420c0e1'
all_authors = stored_meta['Author'].str.split(',').tolist()
flattened_authors = [author.strip() for sublist in all_authors for author in sublist]
unique_authors = list(set(flattened_authors))

def extract_author_ner(query):
    doc = nlp(query)
    # Loop over entities to find PERSON entities (likely to be authors)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()  # Return the first person name found
    return None

def fuzzy_match_author(extracted_author, unique_authors, threshold=80):
    # Use fuzzy matching to find the best match
    matches = process.extractBests(extracted_author, unique_authors, score_cutoff=threshold)
    return matches

def retrieve_from_pinecone(query_text: str, top_k=10, author_name: list = None):  
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)  # Update with your API key and environment
    index_name = 'embeddings6'

    # Connect to the existing index
    index = pc.Index(index_name)

    # Embed the query text using the same model used for upserts
    query_embedding = model.encode(query_text).tolist()  # Encode query into embedding

    # Build the metadata filter if an author name is provided
    metadata_filter = None
    if author_name and isinstance(author_name, list):  # Ensure it's a list
        metadata_filter = {
            "Author": {"$in": author_name}  # Filter using the $in operator for the author name(s)
        }

    # Perform similarity search in Pinecone with metadata filtering
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_values=True,
            include_metadata=True,
            filter=metadata_filter  # Apply the metadata filter if it exists
        )
    except Exception as e:
        print(f"Error during Pinecone query: {e}")
        return

    # Display the results
    print(f"Top {top_k} results for the query '{query_text}' with author filtering:")
    for i, match in enumerate(results['matches']):
        print(f"\nResult {i+1}:")
        print(f"  - ID: {match['id']}")
        print(f"  - Score: {match['score']}")
        print(f"  - Metadata: {match.get('metadata', {})}")

    return results

def retrive_from_chatbot(query,results):
    context_text = "\n\n---\n\n".join([doc['metadata']['text'] for doc in results['matches']])
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}
    ---
    Answer the question based on the above context: {question}
    """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    model1 = Ollama(model="llama3.1")
    response_text = model1.invoke(prompt)
    sources ='sources are ' + ', '.join([Author['id'] for Author in results['matches'][0:1]]) + '\n' +'Publication dates: ' + ', '.join(result['metadata']['publication_date']  for result in results['matches'][:1])
    formatted_response = f"Response: {response_text}\n{sources}"
    #print(formatted_response)
    
    return formatted_response
chat_history = []
query_cache = {}
clarification_needed = False  # To keep track if clarification is required
previous_query = ""  # Store the last query
confidence_threshold = 0.6  # Set a threshold for confidence


def chatbot_interface(query):
    global clarification_needed, previous_query
    additional_info = None

    if clarification_needed:
        # Combine previous query with new details
        combined_query = previous_query + " " + query
        clarification_needed = False  # Reset the flag
        additional_info = query  # Store the new information
    else:
        combined_query = query  # Use the current query

    # Check if the combined query has already been asked
    if combined_query in query_cache:
        response_text = query_cache[combined_query]
    else:
        extracted_authors = extract_author_ner(combined_query)

        if extracted_authors:
            matched_authors = fuzzy_match_author(extracted_authors, unique_authors)

            if matched_authors:
                matched_author_names = [match[0] for match in matched_authors if isinstance(match[0], str)]
                results = retrieve_from_pinecone(combined_query, author_name=matched_author_names)
            else:
                results = retrieve_from_pinecone(combined_query)
        else:
            results = retrieve_from_pinecone(combined_query)

        # Check the highest confidence score
        if results and 'matches' in results and len(results['matches']) > 0:
            highest_score = results['matches'][0]['score']
            if highest_score < confidence_threshold:  # Check against the confidence threshold
                response_text = "I'm not confident about my findings. Could you please provide more details?"
                clarification_needed = True  # Set flag to indicate clarification is needed
                previous_query = combined_query  # Store current query for future use
            else:
                response_text = retrive_from_chatbot(combined_query, results, additional_info)
        else:
            response_text = "I'm not sure what you're asking. Could you please provide more details?"
            clarification_needed = True
            previous_query = combined_query

        # Cache the query and its response
        query_cache[combined_query] = response_text

    # Add the response to chat history (without displaying the final query)
    chat_history.append((query, response_text))

    # Return the updated chat history
    return chat_history

def gradio_chat_interface():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()  # Chatbot UI component
        user_input = gr.Textbox(placeholder="Enter your query here...", label="User Query")
        submit_button = gr.Button("Submit")

        def on_submit(query):
            # Call the chatbot interface function to get the response
            updated_history = chatbot_interface(query)
            return updated_history
        
        submit_button.click(on_submit, inputs=user_input, outputs=chatbot)

    demo.launch()
    
gradio_chat_interface()