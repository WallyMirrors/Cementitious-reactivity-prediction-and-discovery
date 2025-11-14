---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: soroush2
    language: python
    name: soroush2
---

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import time

```

```python
import os
from pymongo import MongoClient
import pymongo
from tqdm import tqdm
import re
import pickle
import json
import torch
import time
import random
from langchain.docstore.document import Document
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.embeddings import SentenceTransformerEmbeddings


from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer
from langchain.vectorstores import Chroma
import json
from bson import ObjectId
import tiktoken
import pandas as pd
```

```python

from openai import OpenAI
client = OpenAI(api_key=api_key)


```

```python
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Assuming the Chroma and embedding classes are already defined
# and the OpenAI API client is configured

# Define your models and corresponding directories
models = [
    {"name": "all-mpnet-base-v2", "directory": "data/chroma/etmpnet", "type": "sentence-transformer"},
    {"name": "all-MiniLM-L6-v2", "directory": "data/chroma/etsminilm", "type": "sentence-transformer"},
    {"name": "m3rg-iitd/matscibert", "directory": "data/chroma/etscibert", "type": "huggingface", "device": "cuda"}
]

# Function to initialize embeddings based on model type
def init_embeddings(model_info):
    if model_info["type"] == "sentence-transformer":
        return SentenceTransformerEmbeddings(model_name=model_info["name"])
    elif model_info["type"] == "huggingface":
        model_kwargs = {'device': model_info.get("device", "cuda")}
        return HuggingFaceEmbeddings(model_name=model_info["name"], model_kwargs=model_kwargs)
    else:
        raise ValueError("Unknown model type")

# Function to process documents
def process_docs(docs, client):
    responses = []
    unrelated_count = 0
    for table in tqdm(docs):

        # Extract table details
        caption = table.page_content
        actual_table = table.metadata['table']
        doi = table.metadata['doi']

        # Query GPT-3.5 Turbo
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Check if the table below contains chemical compositions or oxides of materials, such as sio2, al2o3, fe2o3 content. If it is related, respond with 1 only, if not, respond with 0 only."},
                {"role": "user", "content": caption}
            ],
            temperature=0.1,
            max_tokens=600,
            top_p=1
        )

        # Process response
        RR = response.choices[0].message.content


        responses.append((RR, caption, doi))

    return responses


def save_responses(model_name, responses):
    model_name = model_name.replace("/", "_").replace("\\", "_")
    
    filename = f"responses_{model_name}.json"
    with open(filename, 'w') as file:
        json.dump(responses, file)
    print(f"Responses saved for model {model_name}")
    
# Main loop
for model_info in models:
    print(f"Processing with model: {model_info['name']}")

    # Initialize embeddings and Chroma DB
    embeddings = init_embeddings(model_info)
    tvdb = Chroma(persist_directory=model_info["directory"], embedding_function=embeddings)

    # Perform similarity search
    question = "What are the major oxide or chemical compositions, such as CaO, SiO2, MgO, Al2O3, of materials?"
    docs = tvdb.similarity_search(question, k=1000)

    # Process documents
    responses = process_docs(docs, client)

    # You can add code here to analyze the responses and evaluate the model's performance

    print(f"Completed processing for model: {model_info['name']}")
    
    # Save responses
    save_responses(model_info['name'], responses)
```

```python
import json
import numpy as np

def get_tp_fp_fn(responses, ground_truth):
    """
    Calculate True Positives, False Positives, and False Negatives.

    :param responses: List of tables retrieved by the embedding model.
    :param ground_truth: Dictionary mapping table captions to their relevance (True or False).
    :return: Tuple (TP, FP, FN)
    """
    TP = FP = FN = 0

    for _, caption, _ in responses:
        is_relevant = ground_truth.get(caption, False)

        if is_relevant:
            TP += 1  # True Positive: Relevant table retrieved
        else:
            FP += 1  # False Positive: Irrelevant table retrieved
    
    FN = sum(ground_truth.values()) - TP  # False Negative: Relevant tables missed by the model

    return TP, FP, FN

def get_relevance_scores(responses):
    """
    Extract relevance scores from responses.

    :param responses: List of responses
    :return: List of relevance scores (1 or 0)
    """
    return [1 if response.strip() == '1' else 0 for response, _, _ in responses]


def calculate_dcg(scores):
    return np.sum([((2**rel - 1) / np.log2(idx + 2)) for idx, rel in enumerate(scores)])

def calculate_ndcg(responses, ideal_responses):
    dcg = calculate_dcg(responses)
    idcg = calculate_dcg(ideal_responses)
    return dcg / idcg if idcg > 0 else 0


performance_table = []
def compute_metrics(responses, ground_truth, cutoff):
    TP, FP, FN = get_tp_fp_fn(responses[:cutoff], ground_truth)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    relevance_scores = [1 if response.strip() == '1' else 0 for response, _, _ in responses[:cutoff]]
    ideal_relevance_scores = [1] * len(relevance_scores)
    ndcg = calculate_ndcg(relevance_scores, ideal_relevance_scores)
    cg = sum(relevance_scores)
    return precision, recall, f1_score, cg, ndcg


# Main loop
performance_table = []
for model_info in models:
    model_name = model_info['name'].replace("/", "_").replace("\\", "_")
    # Load responses
    filename = f"responses_{model_name}.json"
    with open(filename, 'r') as file:
        responses = json.load(file)
        
    ground_truth = {caption: response.strip() == '1' for response, caption, _ in responses}

    for cutoff in [10, 100, 1000]:
        precision, recall, f1_score, cg, ndcg = compute_metrics(responses, ground_truth, cutoff)
        performance_table.append((model_name, cutoff, precision, recall, f1_score, cg, ndcg))

```

```python

# Create DataFrame
df = pd.DataFrame(performance_table, columns=['Model', 'Cutoff', 'Precision', 'Recall', 'F1 Score', 'CG','NDCG'])

# Format the DataFrame for better readability (optional)
df['Precision'] = df['Precision'].map('{:.2f}'.format)
df['Recall'] = df['Recall'].map('{:.2f}'.format)
df['NDCG'] = df['NDCG'].map('{:.2f}'.format)
df['F1 Score'] = df['F1 Score'].map('{:.2f}'.format)

print(df)
```

```python
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Assuming the Chroma and embedding classes are already defined
# and the OpenAI API client is configured
def get_relevance_scores(response):
    """
    Extract relevance scores from responses.

    :param responses: List of responses
    :return: List of relevance scores (1 or 0)
    """
    return 1 if response.strip() == '1' else 0

# Define your models and corresponding directories
models = [
    {"name": "all-mpnet-base-v2", "directory": "data/chroma/etmpnet", "type": "sentence-transformer"},
    {"name": "all-MiniLM-L6-v2", "directory": "data/chroma/etsminilm", "type": "sentence-transformer"},
    {"name": "m3rg-iitd/matscibert", "directory": "data/chroma/etscibert", "type": "huggingface", "device": "cuda"}
]

# Function to initialize embeddings based on model type
def init_embeddings(model_info):
    if model_info["type"] == "sentence-transformer":
        return SentenceTransformerEmbeddings(model_name=model_info["name"])
    elif model_info["type"] == "huggingface":
        model_kwargs = {'device': model_info.get("device", "cuda")}
        return HuggingFaceEmbeddings(model_name=model_info["name"], model_kwargs=model_kwargs)
    else:
        raise ValueError("Unknown model type")

# Function to process documents
def process_docs(docs, client):
    responses = []
    unrelated_count = 0
    for table in tqdm(docs):
        try:
            # Extract table details
            caption = table.page_content
            actual_table = table.metadata['table']
            doi = table.metadata['doi']

            # Query GPT-3.5 Turbo
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Check if the table below contains chemical compositions or oxides of materials, such as sio2, al2o3, fe2o3 content. If it is related, respond with 1 only, if not, respond with 0 only."},
                    {"role": "user", "content": f"{caption} {actual_table}"}
                ],
                temperature=0.1,
                max_tokens=600,
                top_p=1
            )
            
            print(actual_table)
            # Process response
            RR = response.choices[0].message.content
            print(RR)

            RR = get_relevance_scores(RR) 
            print(RR)

            responses.append((RR, caption, doi))
        except Exception as e:
            print(e)
            responses.append((0, caption, doi))
    return responses


def save_responses(model_name, responses):
    model_name = model_name.replace("/", "_").replace("\\", "_")
    
    filename = f"responses_{model_name}.json"
    with open(filename, 'w') as file:
        json.dump(responses, file)
    print(f"Responses saved for model {model_name}")
    
# Main loop
for model_info in models:
    print(f"Processing with model: {model_info['name']}")

    # Initialize embeddings and Chroma DB
    embeddings = init_embeddings(model_info)
    tvdb = Chroma(persist_directory=model_info["directory"], embedding_function=embeddings)

    # Perform similarity search
    question = "What are the major oxide or chemical compositions, such as CaO, SiO2, MgO, Al2O3, of materials?"
    docs = tvdb.similarity_search(question, k=100000)
    sampled_docs = docs[::50]  # This will take every 50th document from the list

    # Process documents
    responses = process_docs(sampled_docs, client)

    # You can add code here to analyze the responses and evaluate the model's performance

    print(f"Completed processing for model: {model_info['name']}_sampled")

    # Save responses
    save_responses(model_info['name']+'_sampled', responses)
        

```

```python
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Assuming the Chroma and embedding classes are already defined
# and the OpenAI API client is configured
def get_relevance_scores(response):
    """
    Extract relevance scores from responses.

    :param responses: List of responses
    :return: List of relevance scores (1 or 0)
    """
    return [1 if response.strip() == '1' else 0]

# Define your models and corresponding directories
model = [
    {"name": "No embedding", "directory": "", "type": ""},
]

# Function to initialize embeddings based on model type
def init_embeddings(model_info):
    if model_info["type"] == "sentence-transformer":
        return SentenceTransformerEmbeddings(model_name=model_info["name"])
    elif model_info["type"] == "huggingface":
        model_kwargs = {'device': model_info.get("device", "cuda")}
        return HuggingFaceEmbeddings(model_name=model_info["name"], model_kwargs=model_kwargs)
    else:
        raise ValueError("Unknown model type")

# Function to process documents
def process_docs(docs, client):
    responses = []
    unrelated_count = 0
    for table in tqdm(docs):
        try:
            # Extract table details
            caption = table.page_content
            actual_table = table.metadata['table']
            doi = table.metadata['doi']

            # Query GPT-3.5 Turbo
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Check if the table below contains chemical compositions or oxides of materials, such as sio2, al2o3, fe2o3 content. If it is related, respond with 1 only, if not, respond with 0 only."},
                    {"role": "user", "content": f"{caption} {actual_table}"}
                ],
                temperature=0.1,
                max_tokens=10,
                top_p=1
            )

            # Process response
            RR = response.choices[0].message.content
            RR = get_relevance_scores(RR) 
            responses.append((RR, caption, doi))
        except Exception as e:
            responses.append((0, caption, doi))
    return responses


def save_responses(model_name, responses):
    model_name = model_name.replace("/", "_").replace("\\", "_")
    
    filename = f"responses_{model_name}.json"
    with open(filename, 'w') as file:
        json.dump(responses, file)
    print(f"Responses saved for model {model_name}")
    
# Main loop
for model_info in model:
    embeddings = init_embeddings(model_info)

    # Initialize embeddings and Chroma DB
    tvdb = Chroma(persist_directory=model_info["directory"], embedding_function=embeddings)

    # Perform similarity search
    question = "What are the major oxide or chemical compositions, such as CaO, SiO2, MgO, Al2O3, of materials?"
    docs = tvdb.similarity_search(question, k=100000)
    sampled_docs = random.sample(docs, 2000)

    # Process documents
    responses = process_docs(sampled_docs, client)

    # You can add code here to analyze the responses and evaluate the model's performance

    print(f"Completed processing for model: {model_info['name']}_sampled")

    # Save responses
    save_responses('No embedding'+'_sampled', responses)
        

```

```python
import numpy as np
import matplotlib.pyplot as plt

keywords = {"al2o3", "fe2o3", "cao", "mgo", "so3", "tio2", "mno", "k2o", "na2o", "loi", "loss of ignition", "ignition loss"}
responses_cleaned = []



  

        
def plot_cumulative_ones(responses, model_name):
    # Extract relevance scores from responses
    relevance_scores = []
    for (score, table, doi) in responses:
        # Check if score is a list
        if isinstance(score, list):
            # Further check if the list is non-empty and contains an integer
            if score and isinstance(score[0], int):
                relevance_scores.append(score[0])
            else:
                print("Found a list, but it's either empty or does not contain an integer:", score)
                relevance_scores.append(0)
        # Check if score is an integer
        elif isinstance(score, int):
            relevance_scores.append(score)
        else:
            print("Score is neither a list nor an integer:", score)
            relevance_scores.append(0)
    


    
    
    # Calculate cumulative sum
    cumulative_ones = np.cumsum(relevance_scores)
    # Prepare x-values for plotting (every 50th document)
    x_values = np.arange(0, len(cumulative_ones) * 50, 50)
    
    # Plot cumulative counts
    plt.plot(x_values, cumulative_ones*50, label=model_name,marker='o',markersize=2)

# Loop through the models and plot
plt.figure(figsize=(12, 6))


# Define your models and corresponding directories
models = [
    {"name": "all-mpnet-base-v2", "directory": "data/chroma/etmpnet", "type": "sentence-transformer"},
    {"name": "all-MiniLM-L6-v2", "directory": "data/chroma/etsminilm", "type": "sentence-transformer"},
    {"name": "m3rg-iitd/matscibert", "directory": "data/chroma/etscibert", "type": "huggingface", "device": "cuda"},
    {"name": "No embedding", "directory": "", "type": ""},
    #{"name": "Okapi BM25 + all-MiniLM-L6-v2", "directory": "", "type": ""},

]

for model_info in models:
    model_name = model_info['name'].replace("/", "_").replace("\\", "_")
    filename = f"responses_{model_name}_sampled.json"

    if os.path.exists(filename):
        with open(filename, 'r') as file:
            responses = json.load(file)
        for score, table, doi in responses:
        # Check if any keyword is in table (case insensitive)
            if any(keyword in str(table).lower() for keyword in keywords):
                responses_cleaned.append((score, table, doi))

        # Determine the number of elements to pad
        num_to_pad = len(responses) - len(responses_cleaned)

        # Create padding tuples, here using (None, None, None) as placeholders
        padding = [(0, 0, 0) for _ in range(num_to_pad)]

        # Extend responses_cleaned with the padding
        responses_cleaned.extend(padding)        
        
        plot_cumulative_ones(responses, model_info['name'])
        if model_info['name']=="No embedding":
            plot_cumulative_ones(responses_cleaned, "Keyword filtering")
    else:
        print(f"File {filename} does not exist.")

        
        
        


        
        
plt.xlim(0, 100000)
plt.ylim(0, 10000)
plt.xlabel('Top-k value')
plt.ylabel('Sampled cumulative gain')
plt.legend(loc=2)
plt.grid(True)

plt.savefig('filename.pdf')
!pdftoppm -png -r 300 filename.pdf filename


plt.show()

```
