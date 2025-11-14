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

from pymongo import MongoClient
from tqdm import tqdm
import re
import pickle
import json


```

```python
client = MongoClient() # intialize client
```

```python
## Papers
all_records = client.predsynth.papers_v2.find()
client.predsynth.papers_v2.estimated_document_count() # Check number of papers in papers collection
```

```python
# Define keywords for cement and concrete
include_keywords = ["cement", "concrete", "cementitious", "mortar", "binder"]

```

```python
def filter_papers_by_criteria(records, include_keywords, exclude_keyword, min_abstract_length):
    filtered_records = []

    # Convert include_keywords to lowercase for case-insensitive matching
    include_keywords = [keyword.lower() for keyword in include_keywords]

    for record in tqdm(records, desc="Filtering records"):
        # Ensure that title and abstract are strings and convert them to lowercase
        title = str(record.get("title", "")).lower()
        abstract = str(record.get("abstract", "")).lower()

        # Check if keywords exist and are a list, then convert to lowercase
        record_keywords = record.get("keywords")
        if record_keywords is None:
            record_keywords = []
        else:
            record_keywords = [keyword.lower() for keyword in record_keywords]

        # Check title, keywords, and abstract for any of the include_keywords
        if (any(re.search(r'\b' + re.escape(keyword) + r'\b', title) for keyword in include_keywords) or
            any(keyword in record_keywords for keyword in include_keywords) or
            any(re.search(r'\b' + re.escape(keyword) + r'\b', abstract) for keyword in include_keywords)):
            # Exclude if it contains the exclude keyword
            if not re.search(r'\b' + re.escape(exclude_keyword) + r'\b', abstract):
                filtered_records.append(record)
    return filtered_records



# Keywords for title and the must-have keywords for abstract
exclude_keyword = 'dental'
min_abstract_length=20
# Apply the function
filtered_records = filter_papers_by_criteria(all_records, include_keywords, exclude_keyword, min_abstract_length)
```

```python
# Count the number of papers
num_related_papers = len(filtered_records)
print(num_related_papers)
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

# Usage
with open('filtered_records.json', 'w') as file:
    json.dump(filtered_records, file, default=json_serialize)
```
