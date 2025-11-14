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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

```python
from sklearn.preprocessing import LabelEncoder
import csv
from tqdm import tqdm
import re
import pickle
import json
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers import AutoModel, AutoTokenizer
from langchain.vectorstores import Chroma
import json
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AdamW
import numpy as np
import random
from IPython.display import HTML, display
import getpass
import logging
import os
import yaml
from datasets import load_dataset
import numpy as np; np.random.seed(123)
import pandas as pd
from tqdm import tqdm
from IPython.display import HTML, display
import json
from langchain import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import pandas as pd
from openai import OpenAI
from ludwig.api import LudwigModel
import torch
import time
```

```python
os.getpid()
```

```python
gt = 0
gt= pd.read_csv('GT_TE_test.csv')
gt.head()

```

```python
client = OpenAI(api_key=api_key)

def Response_openai(Table,model_name):
    results = []
    for Table in tqdm(Tables):

                # Check if this caption and DOI already exist in the existing DataFrame
        response = client.chat.completions.create(
        model=model_name, messages=[
        {"role": "system", "content": "You are an assistant trained to parse unstructured HTML tables into CSV format."},
        {"role": "user", "content": f"""CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
        Handling Missing Data: Use 'null' for any missing data in these columns.
        Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.

        Unstructured HTML table: {Table}"""}],
        temperature=0.0,
        max_tokens=500)

        Response=response.choices[0].message.content
        results.append(Response)

    return results
    

def clear_cache():
    model = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
def Response_ludwig(Table,model_name):
    clear_cache()
    model = LudwigModel.load(model_name)
    
    results = []
    for Table in tqdm(Tables):

        Prompt = (f"""<s>[INST]Primary Task:Parse an unstructured HTML table and reformat it into a CSV file.
        CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
        Handling Missing Data: Use 'NaN' for any missing data in these columns.
        Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.


        Unstructured HTML table: {Table}
        [INST]""")

        ddf = pd.DataFrame({'Prompt': [Prompt]})
        prediction, _ = model.predict(ddf)
        result = prediction.Winner_response
        results.append(result[0])
    return results

def Response_hf(Tables,model_name):
    clear_cache()

    model = None
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True, device_map="auto"
    )

    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.max_new_tokens = 500
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

    model = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"device":'cuda'})
    
    results = []
    for Table in tqdm(Tables):
        # Task-specific prompt
        prompt_template = (
            f"""[INST] <<SYS>>\
            You are a helpful assistant. Always answer to the point and be as helpful as possible.  

            <</SYS>>

            Task: When presented with a table, particularly those detailing major oxides and chemical compositions (e.g., SiO2, Al2O3), your task is to reformat the data into a structured CSV format. The expected columns are: Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI. For any missing data in these columns, use 'NaN'. If the table does not relate to chemical compositions, simply respond with 'NR'. Do not change the column names, and always have the specific columns mentioned in the examples below.

            Examples for Reference:

            TABLE:
            Chemical composition of the starting materials
            [['Oxides', 'Raw dolomite', 'Calcined alumina'], ['SiO2', '0.51', '0.02'], ['Fe2O3', '0.27', '0.02'], ['Al2O3', '0.54', '99.50'], ['CaO', '31.32', '-'], ['MgO', '20.16', '-'], ['L.O.I', '46.82', '0.11']]

            RESPONSE:
            Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI
            Raw dolomite, 0.51, 0.54, 0.27, 31.32, 20.16, NaN, NaN, NaN, NaN, NaN, 46.82
            Calcined alumina, 0.02, 99.50, 0.02, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.11

            TABLE:
            Oxide composition of BA
            [['Oxide', 'LOI', 'SiO2', 'CaO', 'Al2O3', 'Fe2O3', 'MgO', 'SO3'], ['Composition (%)', '6.90', '63.16', '8.40', '9.70', '5.40', '2.90', '2.87']]

            RESPONSE:
            Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI
            BA, 63.16, 9.70, 5.40, 8.40, 2.90, 2.87, NaN, NaN, NaN, NaN, 6.90

            TABLE:
            Porosity and true density of the clay and the olive pruning waste./n[['Raw materials', 'True density rtrue  [kg/m3]', 'Porosity n [-]'], ['Olive waste', '1251', '0,23'], ['Clay', '2859', '0,37']]

            RESPONSE: NR

            \n\n Your Task:
            TABLE: {Table}

            RESPONSE:
            [/INST]"""

        )
        res = model(prompt_template)
        results.append(res)
    model = None
    return results



```

```python
import io

    
def parse_response_to_df(response, expected_columns):
    # Clean the response string

    cleaned_response = str(response).replace("null","NaN")
    cleaned_response = ''.join(map(str, cleaned_response))
    cleaned_response = cleaned_response.strip('[').strip(']').strip('""').strip('''''').strip('\'"').replace('\\n', '\n').replace('\\', '')

    # Convert expected columns to lower case
    expected_columns_lower = [col.lower() for col in expected_columns]
    
    # Initialize a new DataFrame with lowercased expected columns
    new_df = pd.DataFrame(columns=expected_columns_lower)

    try:
        # Convert the cleaned response string to a DataFrame
        df = pd.read_csv(io.StringIO(cleaned_response))
        # Convert all column names in df to lower case for comparison
        df.columns = [col.lower() for col in df.columns]
        df.columns = df.columns.str.strip()

        # Copy data from original DataFrame or fill with NaN
        for exp_col in expected_columns_lower:
            if exp_col in df.columns:
                new_df[exp_col] = df[exp_col]
            else:
                new_df[exp_col] = pd.NA  # Use pd.NA for better handling of missing data
    except Exception as e:
        print(e)
    return new_df


expected_columns = ['Material Name', 'SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'TiO2', 'MnO', 'K2O', 'Na2O', 'LOI']
expected_columns_lower = [col.lower() for col in expected_columns]

```

```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error


def check_relevance(resp):
    # Check if response is 'NR' which means Not Related
    if isinstance(resp,list): resp = resp[0]
        
        
    resp = str(resp).strip('""/\\[]')    
        
    if resp.lower() == 'nr':
        return 0  # Indicating non-relevant

    else: 
        return 1


# Functions to calculate metrics
def calculate_accuracy(true_labels, predicted_labels):

    return accuracy_score(true_labels, predicted_labels)

def calculate_precision_recall_f1(true_labels, predicted_labels):

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels)
    return precision[0], recall[0], f1[0]

def calculate_rmse(true_values, predicted_values):

    return np.sqrt(mean_squared_error(true_values, predicted_values))
def safe_lower(x):

    try:
        return x.lower()
    except AttributeError:
        return x  # If x is not a string, return it as is
```

```python
All_models = [
    #{"Name": "ft:gpt-3.5-turbo-1106:personal::8faiLnzJ", "Type": "openai"},
    #{"Name": "gpt-3.5-turbo", "Type": "openai"},
    #{"Name": "gpt-4", "Type": "openai"},
    #{"Name": "QLORA_mistral_TE_CC", "Type": "Ludwig"},
    #{"Name": "QLORA_llama2_TE_CC", "Type": "Ludwig"},
    #{"Name": "mistralai/Mistral-7B-Instruct-v0.2", "Type": "hf"},
    {"Name": "meta-llama/Llama-2-7b-hf", "Type": "hf"}
]



Tables = []
Winners = []
for _, (Table, Winner) in gt.iterrows():
    Tables.append(Table)
    Winners.append(Winner.replace('null','NaN'))

```

```python
def responses_all(model_info,Tables,Winners):
    
    
    model_name = model_info['Name']
    model_type = model_info['Type']

    t = time.time()
    if model_type == "openai": responses = Response_openai(Tables, model_name)
    if model_type == "Ludwig": responses = Response_ludwig(Tables, model_name)
    if model_type == "hf": responses = Response_hf(Tables, model_name)
    elapsed = time.time() - t
        
    return responses, np.divide(elapsed,len(responses))
    
    
def metrics_all(Winners, Responses):
    
    metrics = []

    for i, Winner in enumerate(Winners):
        
        relevance_true, relevance_pred, acc_rel, prec_rel, recall_rel, f1_rel, acc_mat, prec_mat, recall_mat, f1_mat, acc_num, rec_num, recall_num, f1_num, rmse,materials_true,materials_pred = ['n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n']

        Response = Responses[i]
        

        # Relevance Detection
        relevance_true = check_relevance(Winner)  # Assuming winner indicates ground truth relevance
        relevance_pred = check_relevance(Response)  # Assuming winner indicates ground truth relevance

        acc_rel = calculate_accuracy([relevance_true], [relevance_pred])

        prec_rel, recall_rel, f1_rel = calculate_precision_recall_f1([relevance_true], [relevance_pred])

        if relevance_true and relevance_pred:

            response_df = parse_response_to_df(Response, expected_columns)
            winner_df = parse_response_to_df(Winner, expected_columns)


            for col in expected_columns_lower[1:]:
                    # Apply to_numeric with errors='coerce' to each column.
                    # This converts non-numeric values to NaN.
                    response_df[col] = pd.to_numeric(response_df[col], errors='coerce')

            for col in expected_columns_lower[1:]:
                    # Apply to_numeric with errors='coerce' to each column.
                    # This converts non-numeric values to NaN.
                    winner_df[col] = pd.to_numeric(winner_df[col], errors='coerce')


            # Inside the evaluate_model function or wherever you're processing materials
            materials_true = [safe_lower(x) for x in winner_df['material name'].tolist()]
            materials_pred = [safe_lower(x) for x in response_df['material name'].tolist()]

            all_materials = materials_true + materials_pred
            
            

            label_encoder = None

            # Create a label encoder and fit it to all possible material names
            label_encoder = LabelEncoder()
            all_materials = materials_true + materials_pred
            label_encoder.fit(all_materials)

            # Transform material names to integer labels
            encoded_materials_true = label_encoder.transform(materials_true)
            encoded_materials_pred = label_encoder.transform(materials_pred)

            # Zero-padding for the shorter list
            max_length = max(len(encoded_materials_true), len(encoded_materials_pred))
            encoded_materials_true = np.pad(encoded_materials_true, (0, max_length - len(encoded_materials_true)), mode='constant')
            encoded_materials_pred = np.pad(encoded_materials_pred, (0, max_length - len(encoded_materials_pred)), mode='constant')

            # Calculate accuracy, precision, recall, and F1 score
            acc_mat = calculate_accuracy(encoded_materials_true.tolist(), encoded_materials_pred.tolist())
            prec_mat, recall_mat, f1_mat = calculate_precision_recall_f1(encoded_materials_true.tolist(), encoded_materials_pred.tolist())

            # Your code for numerical columns and RMSE
            numerical_columns = [col for col in expected_columns_lower if col != 'material name']



            true_values = winner_df[numerical_columns].fillna(0).values.flatten().tolist()
            predicted_values = response_df[numerical_columns].fillna(0).values.flatten().tolist()





            # Zero-padding for the shorter list
            max_length = max(len(true_values), len(predicted_values))
            true_values = np.pad(true_values, (0, max_length - len(true_values)), mode='constant').tolist()
            predicted_values = np.pad(predicted_values, (0, max_length - len(predicted_values)), mode='constant').tolist()

            label_encoder = None
            label_encoder = LabelEncoder()
            all_numbers = true_values+predicted_values
            label_encoder.fit(all_numbers)

            # Transform material names to integer labels
            encoded_true_values = label_encoder.transform(true_values)
            encoded_predicted_values = label_encoder.transform(predicted_values)

            acc_num = calculate_accuracy(encoded_true_values.tolist(), encoded_predicted_values.tolist())

            rec_num, recall_num, f1_num = calculate_precision_recall_f1(encoded_true_values.tolist(), encoded_predicted_values.tolist())

            rmse = calculate_rmse(true_values, predicted_values)         

                

        metrics.append([relevance_true, relevance_pred, acc_rel, prec_rel, recall_rel, f1_rel, acc_mat, prec_mat, recall_mat, f1_mat, acc_num, rec_num, recall_num, f1_num, rmse,materials_true,materials_pred])

    return metrics
```

```python



for model_info in All_models:
    
    print(model_info['Name'])

    Responses, elapsed_avg = responses_all(model_info,Tables,Winners)
    metrics = metrics_all(Winners, Responses)
    print(metrics)    
    # Save elapsed_avg, Responses, and metrics using pickle
    pickle_data = {
        'elapsed_avg': elapsed_avg,
        'Responses': Responses,
        'metrics': metrics
    }

    with open('data' + str(model_info['Name']).replace("/","") + '-test.pkl', 'wb') as file:
        pickle.dump(pickle_data, file)
```

```python
metrics = metrics_all(Winners, Responses)
print(metrics)    
# Save elapsed_avg, Responses, and metrics using pickle
pickle_data = {
    'elapsed_avg': elapsed_avg,
    'Responses': Responses,
    'metrics': metrics
}

with open('data' + str(model_info['Name']).replace("/","") + '-test.pkl', 'wb') as file:
    pickle.dump(pickle_data, file)
```

```python
def Response_ludwig_openai(Table,model_name_ludwig,model_name_openai):
    results = []
    
    clear_cache()
    model = LudwigModel.load(model_name_ludwig)
    
    for Table in tqdm(Tables):
        
        Prompt = (f"""<s>[INST]Primary Task:Parse an unstructured HTML table and reformat it into a CSV file.
        CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
        Handling Missing Data: Use 'NaN' for any missing data in these columns.
        Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.


        Unstructured HTML table: {Table}
        [INST]""")

        ddf = pd.DataFrame({'Prompt': [Prompt]})
        prediction, _ = model.predict(ddf)
        result = prediction.Winner_response
        
                # Check if this caption and DOI already exist in the existing DataFrame
        response = client.chat.completions.create(
        model=model_name_openai,messages=[
        {"role": "system", "content": "You are an assistant trained to parse unstructured HTML tables into CSV format."},
        {"role": "user", "content": f"""CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
        Handling Missing Data: Use 'null' for any missing data in these columns.
        Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.

        Unstructured HTML table: {Table}"""},
        {"role": "assistant", "content": f"{str(result)}"},
        {"role": "user", "content":"Revise the CSV table and give it back to me. Make sure all the materials in the table were extracted. In addition, all the chemical compositions were extracted. In addition, the sequence of chemical compositions are correct. In addition, the column names is the same as I requested. No explanations and just give me the CSV table."}
        ],
        temperature=0.0,
        max_tokens=500)

        Response=response.choices[0].message.content
        results.append(Response)

    return results
    
def Response_openai_ludwig(Table,model_name_openai, model_name_ludwig):
    results = []
    
    clear_cache()
    model = LudwigModel.load(model_name_ludwig)
    
    for Table in tqdm(Tables):
        

        
                # Check if this caption and DOI already exist in the existing DataFrame
        response = client.chat.completions.create(
        model=model_name_openai,messages=[
        {"role": "system", "content": "You are an assistant trained to parse unstructured HTML tables into CSV format."},
        {"role": "user", "content": f"""CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
        Handling Missing Data: Use 'null' for any missing data in these columns.
        Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.

        Unstructured HTML table: {Table}"""}],
        temperature=0.0,
        max_tokens=500)
        
        Response=response.choices[0].message.content
        
        Prompt = (f"""<s>[INST]Primary Task:Parse an unstructured HTML table and reformat it into a CSV file.
        CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
        Handling Missing Data: Use 'NaN' for any missing data in these columns.
        Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.


        Unstructured HTML table: {Table}
        [/INST]
        {Response}
        </s>[INST]
        Revise the CSV table and give it back to me. Make sure all the materials in the table were extracted. In addition, all the chemical compositions were extracted. In addition, the sequence of chemical compositions are correct. In addition, the column names is the same as I requested. No explanations and just give me the CSV table.
        [/INST]""")

        ddf = pd.DataFrame({'Prompt': [Prompt]})
        prediction, _ = model.predict(ddf)
        result = prediction.Winner_response
        
        results.append(result)

    return results


```

```python

```

```python
def responses_seq(model_info1,model_info2,Tables,Winners):
    
    
    model_name1 = model_info1['Name']
    model_type1 = model_info1['Type']
    
    model_name2 = model_info2['Name']
    model_type2 = model_info2['Type']
    
    t = time.time()
    
    if model_type1 == "openai": responses = Response_openai_ludwig(Tables, model_name1, model_name2)
    if model_type1 == "Ludwig": responses = Response_ludwig_openai(Tables, model_name1, model_name2)

    elapsed = time.time() - t
        
    return responses, np.divide(elapsed,len(responses))

```

```python
seq_models = [
    {"Name": "ft:gpt-3.5-turbo-1106:personal::8faiLnzJ", "Type": "openai"},
    {"Name": "QLORA_mistral_TE_CC", "Type": "Ludwig"},
]

```

```python


for i, model_info in enumerate(seq_models):

    Responses, elapsed_avg = responses_seq(model_info,seq_models[i-1],Tables,Winners)
    
    metrics = metrics_all(Winners, Responses)
    print(metrics)    
    # Save elapsed_avg, Responses, and metrics using pickle
    pickle_data = {
        'elapsed_avg': elapsed_avg,
        'Responses': Responses,
        'metrics': metrics
    }

    with open('data' + str(model_info['Name']).replace("/","") + str(seq_models[i-1]['Name']).replace("/","") + '-test.pkl', 'wb') as file:
        pickle.dump(pickle_data, file)
```

```python
seq_models = [
    {"Name": "ft:gpt-3.5-turbo-1106:personal::8faiLnzJ", "Type": "openai"},
    {"Name": "QLORA_mistral_TE_CC", "Type": "Ludwig"},
]

```

```python


for i, model_info in enumerate(seq_models):

    Responses, elapsed_avg = responses_seq(model_info,seq_models[i-1],Tables,Winners)
    
    metrics = metrics_all(Winners, Responses)
    print(metrics)    
    # Save elapsed_avg, Responses, and metrics using pickle
    pickle_data = {
        'elapsed_avg': elapsed_avg,
        'Responses': Responses,
        'metrics': metrics
    }

    with open('data' + str(model_info['Name']).replace("/","") + str(seq_models[i-1]['Name']).replace("/","") + '-test.pkl', 'wb') as file:
        pickle.dump(pickle_data, file)
```

```python
loaded_elapsed_avg = []
loaded_metrics = []
loaded_Responses = []

All_All_models = [
    {"Name": "ft:gpt-3.5-turbo-1106:personal::8faiLnzJ", "Type": "openai"},
    {"Name": "gpt-3.5-turbo", "Type": "openai"},
    {"Name": "gpt-4", "Type": "openai"},
    {"Name": "QLORA_mistral_TE_CC", "Type": "Ludwig"},
    {"Name": "QLORA_llama2_TE_CC", "Type": "Ludwig"},
    {"Name": "mistralaiMistral-7B-Instruct-v0.2", "Type": "hf"},
    {"Name": "meta-llamaLlama-2-7b-hf", "Type": "hf"},
    {"Name": "QLORA_mistral_TE_CCft:gpt-3.5-turbo-1106:personal::8faiLnzJ", "Type": "n"},
    {"Name": "ft:gpt-3.5-turbo-1106:personal::8faiLnzJQLORA_mistral_TE_CC", "Type": "n"}
    
]




for model_info in All_All_models:


    with open('data' + str(model_info['Name']) + '-test.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    
    elapsed_avg=loaded_data['elapsed_avg']
    Responses=loaded_data['Responses']
    
    metrics = metrics_all(Winners, Responses)
    # Save elapsed_avg, Responses, and metrics using pickle
    pickle_data = {
        'elapsed_avg': elapsed_avg,
        'Responses': Responses,
        'metrics': metrics
    }

    with open('data' + str(model_info['Name']) + '-test.pkl', 'wb') as file:
        pickle.dump(pickle_data, file)
        
        
```

```python
for model_info in All_All_models:


    with open('data' + str(model_info['Name']) + '-test.pkl', 'rb') as file:
        loaded_data = pickle.load(file)

    # Access the data
    loaded_elapsed_avg.append(loaded_data['elapsed_avg'])
    loaded_Responses.append(loaded_data['Responses'])
    loaded_metrics.append([str(model_info['Name']),loaded_data['metrics']])
```

```python
len(loaded_metrics)
```

```python

all_data=[]
# Iterate through each model and its metrics
for model_name, metrics in loaded_metrics:
    # Process each metric entry
    for metric in metrics:
        # Extract the first 11 elements
        metric_data = metric[:15]

        # Add the model name and the metric data to all_data
        all_data.append([model_name] + metric_data)

# Define column names for the DataFrame
columns = ["Model Name", "Metric 1", "Metric 2", "Metric 3", "Metric 4", "Metric 5",
           "Metric 6", "Metric 7", "Metric 8", "Metric 9", "Metric 10", "Metric 11","Metric 12", "Metric 13", "Metric 14", "Metric 15"]

# Create a DataFrame from the collected data
df = pd.DataFrame(all_data, columns=columns)
# Iterate over all columns except the first one
for column in df.columns[1:]:
    # Convert to numeric, coercing errors (which turns non-numeric strings to NaN)
    df[column] = pd.to_numeric(df[column], errors='coerce')
df.head()
```

```python
df_copy = df.copy()
df_copy.drop(['Metric 1', 'Metric 2'], axis=1, inplace=True)

```

```python
# Rename columns
new_column_names = {
    'Metric 1': 'Relevance - True',
    'Metric 2': 'Relevance - Pred',    
    'Metric 3': 'Relevance - Binary accuracy',
    'Metric 4': 'Relevance - Precision',
    'Metric 5': 'Relevance - Recall',
    'Metric 6': 'Relevance - F1',
    'Metric 7': 'Comprehensiveness - Multiclass accuracy',
    'Metric 8': 'Comprehensiveness - Precision',
    'Metric 9': 'Comprehensiveness - Recall',
    'Metric 10': 'Comprehensiveness - F1',
    'Metric 11': 'Factuality - Multiclass accuracy',
    'Metric 12': 'Factuality - Precision',
    'Metric 13': 'Factuality - Recall',
    'Metric 14': 'Factuality - F1',
    'Metric 15': 'Factuality - Root mean squared error',
}
df.rename(columns=new_column_names, inplace=True)

# Display the DataFrame
df.head()

```

```python
df
```

```python

# Group the DataFrame by 'Model Name' and calculate the mean for each group
avg_metrics_df = df_copy.groupby('Model Name').mean()

# Reset the index to make 'Model Name' a column again
avg_metrics_df.reset_index(inplace=True)

avg_metrics_df.rename(columns=new_column_names, inplace=True)

# Display the DataFrame with average metrics
print(avg_metrics_df)
```

```python
avg_metrics_df.to_csv("Allmetrics.csv", sep=',')

```

```python
a = [1.23, 2.24]

label_encoder = LabelEncoder()
all_materials = a
label_encoder.fit(all_materials)

# Transform material names to integer labels
encoded_materials_true = label_encoder.transform(a)
print(encoded_materials_true)
```

```python

```
