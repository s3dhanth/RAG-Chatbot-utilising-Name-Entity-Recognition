from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from fuzzywuzzy import fuzz, process
from pinecone import Pinecone
import spacy
from query_rag import extract_author_ner
from query_rag import fuzzy_match_author
from query_rag import retrieve_from_pinecone
from query_rag import retrive_from_chatbot


nlp = spacy.load("en_core_web_lg")
stored_meta = pd.read_csv('arxiv_metadata.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')
api_key = 'd7204d21-cb62-4544-b49c-9169b420c0e1'
all_authors = stored_meta['Author'].str.split(',').tolist()
flattened_authors = [author.strip() for sublist in all_authors for author in sublist]
unique_authors = list(set(flattened_authors))

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_conclusion_and_future_work():
    assert query_and_validate(
        question="What is the primary focus of the proposed MaMA method?",
        expected_response="he MaMA method focuses on adapting the full CLIP model to mammography by utilizing its multi-view nature, addressing challenges such as labeled data scarcity, high-resolution images with small regions of interest, and data imbalance.")
def test_conclusion_and_future_work2():
    assert query_and_validate(
        question = "What are the significant challenges associated with applying CLIP to mammography?",
        expected_response = "The challenges include the scarcity of labeled data, the high-resolution images with small regions of interest, and the imbalance in the data, where most mammograms do not contain cancer.")
def test_conclusion_and_future_work3():
    assert query_and_validate(
        question = "How does the MaMA framework address the issue of limited clinical reports in mammography?",
        expected_response = "MaMA proposes an intuitive method for template-based report construction from tabular data, which helps enable visual-language pre-training despite the lack of corresponding clinical reports.")
# Example workflow after extracting authors


def query_and_validate(question: str, expected_response: str):
    extracted_authors = extract_author_ner(question)
    if extracted_authors:
        print(f"Extracted Author from query: {extracted_authors}")
        
        # List of authors from vector store metadata (already fetched)
        # unique_authors is assumed to be defined

        # Perform fuzzy matching to find the closest match
        matched_authors = fuzzy_match_author(extracted_authors, unique_authors)

        print(f"Matched authors output: {matched_authors}")  # Debugging print statement
        print(f"Type of matched_authors: {type(matched_authors)}")

        if matched_authors:
            # Ensure only author names (strings) are extracted
            matched_author_names = [match[0] for match in matched_authors if isinstance(match[0], str)]

            print(f"Matched authors (only names): (matched_author_names)")
            results =retrieve_from_pinecone(question, author_name=matched_author_names)
        else:
            results =retrieve_from_pinecone(question)
    else:
        results = retrieve_from_pinecone(question)


    response_text =retrive_from_chatbot(question,results)
        
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="llama3.1")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )