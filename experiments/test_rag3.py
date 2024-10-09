from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from pinecone import Pinecone
import spacy
from query_rag_spector import retrieve_from_pinecone
from query_rag_spector import retrive_from_chatbot

nlp = spacy.load("en_core_web_lg")
stored_meta = pd.read_csv('arxiv_metadata.csv')
model = SentenceTransformer('allenai-specter')
api_key = 'd7204d21-cb62-4544-b49c-9169b420c0e1'

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') (Answer with 'true' or 'false') Does the actual response contain the key information or facts from the expected response? Focus on:
1. Publication date
2. Author name
2. Source
"""



def test_conclusion_and_future_workr():
    assert query_and_validate(
        question = "only give the title of paper where one of the author of paper is Wenliang Zhao or Minglei Shi",
        expected_response = "Source : file:///C:/QpiAi/paper_2409.18128.pdf, Author : Wenliang Zhao, Minglei Shi, Xumin Yu, Jie Zhou, Jiwen Lu")


def test_conclusion_and_future_workr1():
    assert query_and_validate(
        question = " What technique is used in FlowTurbo to reduce computational costs?",
        expected_response = " Source : file:///C:/QpiAi/paper_2409.18128.pdf, Author : Wenliang Zhao, Minglei Shi, Xumin Yu, Jie Zhou, Jiwen Lu")


def test_conclusion_and_future_workr4():
    assert query_and_validate(
        question = " How does the velocity predictor behave during the sampling process in flow-based models?",
        expected_response = " Source : file:///C:/QpiAi/paper_2409.18128.pdf, Author : Wenliang Zhao, Minglei Shi, Xumin Yu, Jie Zhou, Jiwen Lu")
    

def test_conclusion_and_future_workr7():
    assert query_and_validate(
        question = "What is the best FID score and the corresponding latency achieved by FlowTurbo in class-conditional image generation experiments?",
        expected_response = " Source : file:///C:/QpiAi/paper_2409.18128.pdf, Author : Wenliang Zhao, Minglei Shi, Xumin Yu, Jie Zhou, Jiwen Lu")

def test_conclusion_and_future_workr8():
    assert query_and_validate(
        question = "How does the pseudo corrector block affect the sampling speed in the experiments, and what is its contribution to the model’s performance?",
        expected_response = " Source : file:///C:/QpiAi/paper_2409.18128.pdf, Author : Wenliang Zhao, Minglei Shi, Xumin Yu, Jie Zhou, Jiwen Lu")
def test_conclusion_and_future_workr9():
    assert query_and_validate(
        question = "what is the report confidence interval of proposed method by bootstrap sampling test set results from linear classification.",
        expected_response = " Source : file:///C:/QpiAi/paper_2409.18119.pdf, Authors :Yuexi Du1, John Onofrey, Nicha C. Dvornek")
    
def test_conclusion_and_future_workr10():
    assert query_and_validate(
        question = "What was the sample size N of bootstrap evaluation for the linear classification predicted result of our method on both BI-RADS and density prediction tasks",
        expected_response = "Source : file:///C:/QpiAi/paper_2409.18119.pdf, Authors :Yuexi Du1, John Onofrey, Nicha C. Dvornek")


def query_and_validate(question: str, expected_response: str):
    
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