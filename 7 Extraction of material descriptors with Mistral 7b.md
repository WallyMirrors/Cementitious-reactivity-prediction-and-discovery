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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time

```

```python
import os
import pymongo
import re
import pickle
import json
import torch
import time
import pandas as pd
from tqdm import tqdm

from langchain_community.document_loaders.merge import MergedDataLoader

from langchain.docstore.document import Document
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever


from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer
from langchain.vectorstores import Chroma
import json
from bson import ObjectId

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.chains import LLMChain
```

```python
# To load the sentencevectordb

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'data/chroma/sminilm'

svdb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
print(svdb._collection.count())
```

```python
def clear_cache():
  if torch.cuda.is_available():
    model = None
    torch.cuda.empty_cache()
```

```python
model = None
llm = None
clear_cache()
tokenizer = None
```

```python
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch 
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto"
)
 
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 200
generation_config.temperature = 0.0001
generation_config.top_p = 0.90
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15
 
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    generation_config=generation_config,
)
 
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"device":'cuda'})


```

```python
llm("The first man on the moon was ...")
```

```python

material_name='Q1'


search=f'What are the characteristics, origin and abbreviation of the material named {material_name}?'
table_caption='Chemical composition of major elements, expressed in wt% of oxides, for stone and quicklime materials. Loss on Ignition (LOI).'
doi="10.1016/j.conbuildmat.2017.06.147"

# Load the QA chain
chain = load_qa_chain(llm, chain_type="stuff")


# Document retrieval (adjust as needed for your database)
try: 
    docs = svdb.similarity_search(search, k=50, filter={"doi":doi })

except Exception as e:
    docs = svdb.similarity_search(search, k=20, filter={"doi":doi })


tab = tvdb.similarity_search('', k=1, filter={"doi":doi })

# Document retrieval (adjust as needed for your database)

abstract = avdb.similarity_search("", k=1, filter={"doi":doi })

found_docs =[]

for doc in docs:
    if f'({material_name})' in doc.page_content.split():
        found_docs.append(doc)
    elif f'({material_name}' in doc.page_content.split():
        found_docs.append(doc)
    elif f'{material_name})' in doc.page_content.split():
        found_docs.append(doc)
        
if len(found_docs)<5:
    for doc in docs:
        if material_name in doc.page_content.split():
            found_docs.append(doc)           


if len(found_docs)>10: 
    found_docs = found_docs[:10]

if len(found_docs)==0: 
    found_docs = docs[:10]


# Task-specific prompt

if len(abstract)==1:
    prompt_template = (
        f"""[INST] <<SYS>>\
        You are a helpful, respectful, and honest assistant. Always strive to provide the most helpful and accurate answers. 
        If a question is unclear, nonsensical, or lacks factual coherence, clarify or explain the issue instead of giving an incorrect response.         If no relevant information about the material is available, respond with NaN.\n\n

        <</SYS>>\n\n
        Task: Describe in one sentence the composition or characteristics of the material called {material_name}, using only the provided information. 
        Related Table:\n\n {table_caption}\n{tab[0].metadata['table']} \n\n
        Abstract: {abstract[0].page_content}
        [/INST]"""

    )
else: 
    prompt_template = (
        f"""[INST] <<SYS>>\
        You are a helpful, respectful, and honest assistant. Always strive to provide the most helpful and accurate answers. 
        If a question is unclear, nonsensical, or lacks factual coherence, clarify or explain the issue instead of giving an incorrect response.         If no relevant information about the material is available, respond with NaN.\n\n

        <</SYS>>\n\n
        Task: Describe in characteristics of the material called {material_name}, using only the provided information. One sentence only.
        Material name: {material_name}\n\n
        Related Table:\n\n {table_caption}\n{tab[0].metadata['table']} \n\n
        [/INST]"""

    )
# Create a PromptTemplate object
prompt_template = PromptTemplate(
    template=prompt_template,
    input_variables=["user_question"]
)


chain = LLMChain(llm=llm, prompt=prompt_template,metadata=None)



# Run the chain with the retrieved documents and custom prompt
res = chain.run(input_documents=found_docs, question=material_name)
print(res)

```

```python
docs
```

```python

# Load the file
materials_df  = pd.read_csv('Final_ext_compositions.csv',encoding='cp1252')

# Display the first few rows of the new DataFrame
materials_df.head()

```

```python
def save_responses(responses):
    # Implement the logic to save 'responses'
    # For example, saving to a file:
    filename = f'descriptions.csv'
    # Convert responses to DataFrame and save as CSV
    pd.DataFrame(responses, columns=['Material Name','Response', 'Caption', 'DOI']).to_csv(filename, index=False)
    
# Placeholder for responses
responses = []
m_is=[]
```

```python

for i, row in enumerate(tqdm(materials_df.iterrows(), total=materials_df.shape[0])):
    try:

        material_name = row[1]['Material name']
        doi = row[1]['DOI']
        caption = row[1]['Caption']

        search=f'Define what is this material: {str(material_name)}?'
        table_caption=caption

        # Load the QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Document retrieval (adjust as needed for your database)

        docs = svdb.similarity_search(search, k=20, filter={"doi":str(doi) })

        tab = tvdb.similarity_search(caption, k=1, filter={"doi":str(doi) })

        # Document retrieval (adjust as needed for your database)

        abstract = avdb.similarity_search(search, k=5, filter={"doi":str(doi) })

        found_docs =[]


        for doc in docs:
            if f'({material_name})' in doc.page_content:
                found_docs.append(doc)
            elif f'({material_name}' in doc.page_content:
                found_docs.append(doc)
            elif f'{material_name})' in doc.page_content:
                found_docs.append(doc)

        if len(found_docs)<5:
            for doc in docs:
                if material_name in doc.page_content:
                    found_docs.append(doc)           


        if len(found_docs)>10: 
            found_docs = found_docs[:10]

        if len(found_docs)==0: 
            found_docs = docs[:10]


        exp = ''

        for doc in found_docs:
            exp+=doc.page_content

        abss = ''

        for doc in abstract:
            abss+=doc.page_content    
            
            
        prompt_template = (
            f"""[INST] <<SYS>>\
            As a helpful, respectful, and honest assistant, your primary goal is to provide the most accurate and helpful answers possible. In cases where a question is unclear, nonsensical, or lacks factual coherence, prioritize clarification or explanation over providing an incorrect response. 
            <</SYS>>\n\n
            Describe the material named '{material_name}' in one sentence only. If information about the material is unavailable, respond with 'Not Available (NaN)' and avoid assumptions or discussions about other materials and avoid talking about the chemical composition of the material. Focus on the following details:

            Material Name: {material_name}\n\n
            Related Table: {table_caption}{tab[0].metadata['table']}:\n\n
            {abss} \n
            {exp}
            [/INST]"""

            )
        # Create a PromptTemplate object
        prompt_template = PromptTemplate(
            template=prompt_template,
            input_variables=["user_question"]
        )


        chain = LLMChain(llm=llm, prompt=prompt_template,metadata=None)

        if len(abstract)>=1:
            abstract.append(found_docs)
            found_docs = abstract
        


        # Run the chain with the retrieved documents and custom prompt
        res = chain.run(input_documents=found_docs, question=material_name)
        responses.append((material_name, res,caption,doi))
    except Exception as e:
        m_is.append(i)
    # Check if it's time to save the responses
    if (i + 1) % 50 == 0:
        save_responses(responses)
        
        
```
