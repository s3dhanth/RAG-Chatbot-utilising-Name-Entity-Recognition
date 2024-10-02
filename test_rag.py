from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
from fuzzywuzzy import fuzz, process
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
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""



def test_conclusion_and_future_workr():
    assert query_and_validate(
        question = "only give the title of paper where one of the author of paper is Wenliang Zhao or Minglei Shi",
        expected_response = "FlowTurbo: Towards Real-time Flow-Based Image pyGeneration with Velocity Refiner")

def test_conclusion_and_future_workr1():
    assert query_and_validate(
        question = " What technique is used in FlowTurbo to reduce computational costs?",
        expected_response = "FlowTurbo reduces computational costs by using a lightweight velocity refiner to estimate the velocity during sampling.")

def test_conclusion_and_future_workr2():
    assert query_and_validate(
        question = "How does FlowTurbo perform on class-conditional generation in terms of acceleration?",
        expected_response = "FlowTurbo achieves an acceleration ratio of 53.1%∼58.3% on class-conditional generation")

def test_conclusion_and_future_workr4():
    assert query_and_validate(
        question = " How does the velocity predictor behave during the sampling process in flow-based models?",
        expected_response = "The velocity predictor’s outputs in flow-based models become stable during the sampling process.")
    
def test_conclusion_and_future_workr5():
    assert query_and_validate(
        question = "How does FlowTurbo improve the inference speed compared to Heun’s method, and what percentage of acceleration is observed?",
        expected_response = "FlowTurbo improves the inference speed by 37.2% to 43.1% compared to Heun’s method, while still maintaining better sampling quality. It achieves these results by optimizing the sampling configuration through the use of Heun’s blocks, pseudo corrector blocks, and velocity refiner blocks.")

def test_conclusion_and_future_workr6():
    assert query_and_validate(
        question = " What is the classifier-free guidance scale (CFG) used during class-conditional image generation, and what is its significance?",
        expected_response = "The classifier-free guidance scale (CFG) used is 1.5. CFG is significant because it allows the model to generate higher-quality images by adjusting the balance between conditional and unconditional guidance, which enhances the fidelity of generated images")   

def test_conclusion_and_future_workr7():
    assert query_and_validate(
        question = "What is the best FID score and the corresponding latency achieved by FlowTurbo in class-conditional image generation experiments?",
        expected_response = "The best FID score achieved by FlowTurbo in class-conditional image generation experiments is 3.63, with a corresponding latency of 41.6 milliseconds per image, indicating real-time image generation performance.")

def test_conclusion_and_future_workr8():
    assert query_and_validate(
        question = "How does the pseudo corrector block affect the sampling speed in the experiments, and what is its contribution to the model’s performance?",
        expected_response = "The pseudo corrector block significantly improves the sampling speed while introducing a negligible performance drop. For instance, using six pseudo corrector blocks (H1P6R2 configuration) reduced the latency to 41.6 ms/img, achieving a speed-up of 38.7% compared to the baseline while maintaining competitive FID scores.")

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
