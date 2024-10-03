from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from pinecone import Pinecone
import spacy


nlp = spacy.load("en_core_web_lg")
stored_meta = pd.read_csv('arxiv_metadata.csv')
model = SentenceTransformer('allenai-specter')

# Get the API_KEY
api_key = os.getenv('API_KEY')
stored_meta = pd.read_csv('arxiv_metadata.csv')

# Start the MLflow tracking
mlflow.set_experiment("Chatbot Queries Tracking")
mlflow.set_tracking_url('hit mlflow ui in terminal and paste the url here')



def retrieve_from_pinecone(query_text: str, top_k=10, author_name: list = None):
    pc = Pinecone(api_key=api_key)
    index_name = 'embeddings6'
    index = pc.Index(index_name)
    query_embedding = model.encode(query_text).tolist()
    metadata_filter = None
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_values=True,
            include_metadata=True,
            filter=metadata_filter
        )
    except Exception as e:
        print(f"Error during Pinecone query: {e}")
        return

    print(f"Top {top_k} results for the query '{query_text}'")
    for i, match in enumerate(results['matches']):
        print(f"\nResult {i+1}:")
        print(f"  - ID: {match['id']}")
        print(f"  - Score: {match['score']}")
        print(f"  - Metadata: {match.get('metadata', {})}")

    return results

def retrive_from_chatbot(query, results):
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
    sources = 'Sources are ' + ', '.join([Author['id'] for Author in results['matches'][0:1]]) + '\n' + 'Publication dates: ' + ', '.join(result['metadata']['publication_date'] for result in results['matches'][:1])
    formatted_response = f"Response: {response_text}\n{sources}"

    return formatted_response

chat_history = []
query_cache = {}
clarification_needed = False
previous_query = ""
confidence_threshold = 0.5

def chatbot_interface(query):
    global clarification_needed, previous_query
    with mlflow.start_run():  # Start MLflow run to track the query and response
        if clarification_needed:
            combined_query = previous_query + " " + query
            clarification_needed = False
        else:
            combined_query = query

        if combined_query in query_cache:
            response_text = query_cache[combined_query]
        else:
             results = retrieve_from_pinecone(combined_query)
        
            if results and 'matches' in results and len(results['matches']) > 0:
                highest_score = results['matches'][0]['score']
                if highest_score < confidence_threshold:
                    response_text = "I'm not confident about my findings. Could you please provide more details?"
                    clarification_needed = True
                    previous_query = combined_query
                else:
                    response_text = retrive_from_chatbot(combined_query, results)
            else:
                response_text = "I'm not sure what you're asking. Could you please provide more details?"
                clarification_needed = True
                previous_query = combined_query

            query_cache[combined_query] = response_text

        chat_history.append((combined_query, response_text))

        # Log the query and response to MLflow
        mlflow.log_param("Query", combined_query)
        mlflow.log_param("Clarification_Needed", clarification_needed)
        mlflow.log_metric("Highest_Score", highest_score if results and 'matches' in results and len(results['matches']) > 0 else 0)
        mlflow.log_text(response_text, "Response")

        return chat_history

def gradio_chat_interface():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        user_input = gr.Textbox(placeholder="Enter your query here...", label="User Query")
        submit_button = gr.Button("Submit")

        def on_submit(query):
            updated_history = chatbot_interface(query)
            return updated_history

        submit_button.click(on_submit, inputs=user_input, outputs=chatbot)

    demo.launch()


gradio_chat_interface()
