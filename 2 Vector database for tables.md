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
from pymongo import MongoClient
import pymongo
from tqdm import tqdm
import re
import pickle
import json
import torch

from langchain.docstore.document import Document
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.embeddings import SentenceTransformerEmbeddings


from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import json
from bson import ObjectId

import json
import pandas as pd
from bs4 import BeautifulSoup
```

```python
from bson import ObjectId
import json

def json_serialize(obj):
    """
    JSON serializer for objects not serializable by default json code
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    raise TypeError ("Type %s not serializable" % type(obj))
    
def json_deserialize(obj):
    # Implement deserialization if needed, e.g., converting string back to datetime
    return obj

```

```python

with open('filtered_tables.json', 'r') as file:
    filtered_tables = json.load(file)
```

```python
filtered_tables[1]
```

```python
def is_real_table(table):
    """ Check if the given table is a real table. """
    # Check if table is not empty
    if not table or not table[0]:
        return False

    # Check for meaningful content (not just placeholders)
    for row in table:
        if all(cell in [None, "", 0, "0"] for cell in row):
            return False

    return True

# Count real and not real tables
real_tables = 0
not_real_tables = 0

for item in filtered_tables:
    if is_real_table(item['act_table']):
        real_tables += 1
    else:
        not_real_tables += 1

print(f"Real tables: {real_tables}")
print(f"Not real tables: {not_real_tables}")

```

```python
def process_table_captions(tables):
    all_captions = []
    seen_captions = set()  # To track duplicates along with DOIs

    for table in tqdm(tables, desc="Processing tables"):
        # Extract the DOI and caption from the table record
        paper_doi = table['paper_doi']
        caption = table['caption']
        actual_table = table['act_table']
        # Create a unique key combining DOI and caption to check for duplicates
        unique_key = (paper_doi, caption)

        # Process the caption if it's a string and the unique key is not already seen
        if isinstance(caption, str) and unique_key not in seen_captions:
            seen_captions.add(unique_key)

            # Store the caption along with its DOI
            doc_caption = Document(page_content=caption+'/n'+str(actual_table),
                                   metadata={"table": json.dumps(actual_table), "doi": paper_doi})
            all_captions.append(doc_caption)

    return all_captions

```

```python
processed_tables = process_table_captions(filtered_tables)
```

```python
len(processed_tables)
```

```python
# Count real and not real tables
real_tables = 0
not_real_tables = 0

for item in processed_tables:
    if is_real_table(json.loads(item.metadata['table'])):
        real_tables += 1
    else:
        not_real_tables += 1

print(f"Real tables: {real_tables}")
print(f"Not real tables: {not_real_tables}")
```

```python
json.loads(processed_tables[12000].metadata['table'])
```

```python
# Define metadata field information
metadata_field_info = [
    AttributeInfo(name="table", description="Actual table content", type="list"),
    AttributeInfo(name="doi", description="Digital Object Identifier of the document", type="string"),
]
```

```python
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'data/chroma/etsminilm'

```

```python
svdb = Chroma.from_documents(processed_tables, embedding=embeddings, persist_directory=persist_directory)
```

```python
svdb.persist()
```

```python
question = "major oxides or chemical compositions of materials including CaO, SiO2, MgO, Al2O3"
docs = svdb.similarity_search(
    question,
    k=60)

for doc  in docs: print(doc)
```
