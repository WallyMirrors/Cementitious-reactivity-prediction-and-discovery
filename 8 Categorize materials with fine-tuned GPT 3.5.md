---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: soroush1
    language: python
    name: soroush1
---

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

```python

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
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
import json
import pandas as pd
```

```python
os.getpid()
```

```python
gt= pd.read_csv('Categorization dataset.csv')
gt.head()
len(gt)
```

```python
dataset_c = []
for _, (Material_Name, Caption, DOI, Descriptor, Category) in gt.iterrows():
    Prompt = f"""How do you categorize the material called '{Material_Name}' based on the following information: 
    Caption of the table that described the chemical composition of this material: {Caption}
    Descriptor from another language model: {Descriptor}
    
    Category of the material '{Material_Name}': """

    dataset_c.append({"Prompt": Prompt, "Winner": str(Category)})
    
len(dataset_c)
```

```python
gt.loc[1]
```

```python
fine_tuning_dataset=[]
```

```python
# Reformatting for fine-tuning
fine_tuning_dataset = []
for item in dataset_c:
    fine_tuning_item = {
        "messages": [
            {"role": "system", "content": "You are an assistant trained to classify the materials based on the description provided and your own knowledge about materials. "},
            {"role": "user", "content": json.dumps(item["Prompt"])},
            {"role": "assistant", "content": json.dumps(item["Winner"])}
        ]
    }
    fine_tuning_dataset.append(fine_tuning_item)
    

```

```python
len(fine_tuning_dataset)
```

```python
fine_tuning_dataset[1]
```

```python
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(fine_tuning_dataset, test_size=0.1)
```

```python

with open('fine_tuning_TE_ENR.jsonl', 'w') as file:
    for item in fine_tuning_dataset:
        json.dump(item, file)
        file.write('\n')
        

with open('fine_tuning_TRAIN_TE_ENR.jsonl', 'w') as file:
    for item in train_data:
        json.dump(item, file)
        file.write('\n')
        

with open('fine_tuning_TEST_TE_ENR.jsonl', 'w') as file:
    for item in test_data:
        json.dump(item, file)
        file.write('\n')
```

```python
len(train_data)
```

```python
!curl https://api.openai.com/v1/files \
  -H "Authorization: Bearer api_key" \
  -F "purpose=fine-tune" \
  -F "file=@fine_tuning_TRAIN_TE_ENR.jsonl" 
```

```python
!curl https://api.openai.com/v1/files \
  -H "Authorization: Bearer api_key" \
  -F "purpose=fine-tune" \
  -F "file=@fine_tuning_TEST_TE_ENR.jsonl" 
```

```python
from openai import OpenAI
client = OpenAI(api_key=api_key)

client.fine_tuning.jobs.create(
  training_file="file-n57msb21C1844BMySxGKKeNL", 
  validation_file="file-6jvYmnyirwy4K9Rm6Ay9SZ0U", 
  model="gpt-3.5-turbo-1106", 
  hyperparameters={
    "n_epochs":"auto"
  }
)
```

```python
def save_responses(responses):
    # Implement the logic to save 'responses'
    # For example, saving to a file:
    filename = f'responses_some.csv'
    # Convert responses to DataFrame and save as CSV
    pd.DataFrame(responses, columns=['Material Name','Caption','DOI','Descriptor','Response']).to_csv(filename, index=False)

```

```python

```

```python

from openai import OpenAI
client = OpenAI(api_key=api_key)


```

```python


# Load the data from CSV files
materials_df = pd.read_csv('final_compositions.csv', encoding='cp1252')


materials_df.head()
```

```python
responses = []
```

```python

for i, (index, row) in enumerate(tqdm(materials_df.iterrows(), total=materials_df.shape[0])):
    RR =0
    material_name = row['Material Name']
    doi = row['DOI']
    caption = row['Caption']
    descriptor = row['Descriptor']
    try: 
        
        Prompt = f"""How do you categorize the material called '{material_name}' based on the following information: 
        Caption of the table that described the chemical composition of this material: {caption}
        Descriptor from another language model: {descriptor}

        Category of the material '{material_name}': """

        # Check if this caption and DOI already exist in the existing DataFrame
        response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal::8kzjZwlt",messages=[
        {"role": "system", "content": "You are an assistant trained to classify the materials based on the description provided and your own knowledge about materials."},
        {"role": "user", "content": Prompt}],
        temperature=0.0,
        max_tokens=100,
        top_p=1,)

        RR=response.choices[0].message.content

        # Check if it's time to save the responses
        if (i + 1) % 100 == 0:
            save_responses(responses)


    except Exception as e:
        print(e)
        
    RR=response.choices[0].message.content
    responses.append((material_name, caption, doi,descriptor, RR))
```

```python
save_responses(responses)
```

```python
import pandas as pd

# Load data
data = pd.read_csv('all_materials_extracted_fgpt.csv', encoding='unicode_escape')
oxide_columns = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'TiO2', 'MnO', 'K2O', 'Na2O', 'LOI']

        
# Function to sum oxide values and handle errors
def sum_oxides(row):
    try:
        return row[oxide_columns].astype(float).sum()
    except ValueError:
        return 'error'

# Apply the function to each row
data['Sum'] = data.apply(sum_oxides, axis=1)

# Identify rows with errors
rows_with_errors = data['Sum'] == 'error'

# Extract rows for manual check
data_non_numeric = data[rows_with_errors]

# Save these rows to a new file for manual checking
data_non_numeric.to_excel('manual_check.xlsx', index=False)

# Optionally, remove these rows from the original DataFrame
data = data[~rows_with_errors]
data.to_csv('Final_ext_compositions.csv', index=False)

```

```python

```
