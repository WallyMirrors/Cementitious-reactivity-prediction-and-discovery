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
from nltk.tokenize import sent_tokenize

from langchain.docstore.document import Document
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.embeddings import SentenceTransformerEmbeddings


from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from langchain.vectorstores import Chroma
import json
from bson import ObjectId

```

```python
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
with open('filtered_records.json', 'r') as file:
    filtered_records = json.load(file, object_hook=json_deserialize)
```

```python
def process_dataset(records):
    all_documents = []
    seen_page_contents = set()  # To track duplicates

    for record in tqdm(records, desc="Filtering records"):
        # Exclude titles and records with 'review' or 'survey' in title
        if isinstance(record.get('title', ''), str):
            title = record.get('title', '').lower()
            if 'review' in title or 'survey' in title:
                continue

        # Process abstract
        abstract = record.get('abstract')
        if isinstance(abstract, str):
            for sentence in sent_tokenize(abstract):
                if len(sentence.split()) >= 10 and sentence not in seen_page_contents:
                    seen_page_contents.add(sentence)
                    doc_sentence = Document(page_content=sentence,
                                            metadata={"section": "abstract", "doi": record.get('doi')})
                    all_documents.append(doc_sentence)

        # Process sentences in paragraphs
        for paragraph in record.get('paragraphs', []):
            section_name = paragraph.get('section_name', '').lower()
            text = paragraph.get('text', '')

            # Check for section 'introduction'
            if 'introduction' in section_name:
                continue

            for sentence in sent_tokenize(text):
                # Check for sentence length and duplication
                if len(sentence.split()) >= 10 and sentence not in seen_page_contents:
                    seen_page_contents.add(sentence)
                    doc_sentence = Document(page_content=sentence,
                                            metadata={"section": section_name, "doi": record.get('doi')})
                    all_documents.append(doc_sentence)

    return all_documents
```

```python
processed_documents = process_dataset(filtered_records)
```

```python
len(processed_documents)
```

```python
# Define metadata field information
metadata_field_info = [
    AttributeInfo(name="section", description="Section of the paper", type="string"),
    AttributeInfo(name="doi", description="Digital Object Identifier of the document", type="string"),
]
```

```python
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'data/chroma/sminilm'
```

```python
svdb = Chroma.from_documents(processed_documents, embedding=embeddings, persist_directory=persist_directory)
```

```python
svdb.persist()
```

```python
# To load the vectordb
svdb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
print(svdb._collection.count())
```

```python

```
