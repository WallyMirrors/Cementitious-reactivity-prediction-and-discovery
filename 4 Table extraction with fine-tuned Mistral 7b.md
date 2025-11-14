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

from tqdm import tqdm
import re
import pickle
import json
import torch
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers import AutoModel, AutoTokenizer
from langchain.vectorstores import Chroma
import json
import torch
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
import torch
import yaml
from ludwig.api import LudwigModel
from datasets import load_dataset
import numpy as np; np.random.seed(123)
import pandas as pd
from tqdm import tqdm
from IPython.display import HTML, display
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
import evaluate
import json
```

```python
torch.__version__
```

```python
gt= pd.read_csv('Table dataset.csv')
gt.head()
```

```python
dataset_c = []
for _, (Table, Winner) in gt.iterrows():
    Prompt = f"""<s>[INST]Primary Task:Parse an unstructured HTML table and reformat it into a CSV file.
    CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
    Handling Missing Data: Use 'NaN' for any missing data in these columns.
    Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.
    
  
    Unstructured HTML table: {Table}
    [INST]"""


    dataset_c.append({"Prompt": Prompt, "Winner": Winner})
```

```python
Dataset = pd.DataFrame(dataset_c)
Dataset.head()
```

```python
dataset_c[1]
```

```python
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(dataset_c, test_size=0.13)
train_df= pd.DataFrame(train_data)
test_df= pd.DataFrame(test_data)
```

```python

train_df.head()
```

```python
train_df.loc[3]['Winner']
```

```python
train_df.loc[3]['Prompt']
```

```python
test_df.head()
```

```python
from IPython.display import HTML, display

def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))

get_ipython().events.register('pre_run_cell', set_css)

def clear_cache():
  if torch.cuda.is_available():
    model = None
    torch.cuda.empty_cache()
```

```python
model = None
clear_cache()


qlora_fine_tuning_config = yaml.safe_load(
"""
  model_type: llm
  base_model:   mistralai/Mistral-7B-v0.1


  input_features:
    - name: Prompt
      type: text


  output_features:
    - name: Winner
      type: text


  prompt:
    template: >-

      ### Prompt: {Prompt}

      ### Winner :
  
  quantization:
    bits: 8

  generation:
    temperature: 0.1
    max_new_tokens: 2048
    
  preprocessing:
    split:
       probabilities:
        - 0.9
        - 0.05
        - 0.05
    global_max_sequence_length: 2048
    
  adapter:
    type: lora

  trainer:
    type: finetune
    epochs: 3
    batch_size: 1
    eval_batch_size: 1
    enable_gradient_checkpointing: true
    gradient_accumulation_steps: 16
    learning_rate: 0.0002
    optimizer:
      type: adamw
      params:
        eps: 1.e-8
        betas:
          - 0.9
          - 0.999
        weight_decay: 0
    learning_rate_scheduler:
      warmup_fraction: 0.03
      reduce_on_plateau: 0
  """
  )


new_model = LudwigModel(config=qlora_fine_tuning_config, logging_level=logging.INFO)
results = new_model.train(dataset=Dataset)
```

```python
save_path='QLORA_mistral_TE_CC'
new_model.save(save_path)
```

```python
predictions, _ = new_model.predict(dataset=test_df)

```

```python
results = predictions.Winner_response
print(results)
```

```python

```
