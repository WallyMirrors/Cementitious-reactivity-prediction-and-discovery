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
gt= pd.read_csv('Table dataset.csv')
gt.head()
```

```python
dataset_c = []
for _, (Table, Winner) in gt.iterrows():
    Prompt = f"""CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
    Handling Missing Data: Use 'null' for any missing data in these columns.
    Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.

    Unstructured HTML table: {Table}"""


    dataset_c.append({"Prompt": Prompt, "Winner": str(Winner).replace("NaN","null")})
    
len(dataset_c)
```

```python
gt.loc[1].Winner
```

```python
# Reformatting for fine-tuning
fine_tuning_dataset = []
for item in dataset_c:
    fine_tuning_item = {
        "messages": [
            {"role": "system", "content": "You are an assistant trained to parse unstructured HTML tables into CSV format."},
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

with open('fine_tuning_TE_CC.jsonl', 'w') as file:
    for item in fine_tuning_dataset:
        json.dump(item, file)
        file.write('\n')
        

with open('fine_tuning_TRAIN_TE_CC.jsonl', 'w') as file:
    for item in train_data:
        json.dump(item, file)
        file.write('\n')
        

with open('fine_tuning_TEST_TE_CC.jsonl', 'w') as file:
    for item in test_data:
        json.dump(item, file)
        file.write('\n')
```

```python
len(fine_tuning_dataset)
```

```python
api_key=api_key
```

```python
!curl https://api.openai.com/v1/files \
  -H "Authorization: Bearer ${api_key}" \
  -F "purpose=fine-tune" \
  -F "file=@fine_tuning_TE_CC.jsonl" 
```

```python
!curl https://api.openai.com/v1/files \
  -H r"Authorization: Bearer ${api_key}" \
  -F "purpose=fine-tune" \
  -F "file=@fine_tuning_TRAIN_TE_CC.jsonl" 
```

```python
!curl https://api.openai.com/v1/files \
  -H "Authorization: Bearer ${api_key} \
  -F "purpose=fine-tune" \
  -F "file=@fine_tuning_TEST_TE_CC.jsonl" 
```

```python
from openai import OpenAI
client = OpenAI(api_key=api_key)

client.fine_tuning.jobs.create(
  training_file="file-q0vjmL1s9EIrud1SsSO1vLT7", 
  validation_file="file-XMf29I1HR1H2O7FrQeO4WzmN", 
  model="gpt-3.5-turbo-1106", 
  hyperparameters={
    "n_epochs":"auto"
  }
)
```

```python
#client.fine_tuning.jobs.cancel("ftjob-dKkESY7D9zgJxlIczb1bR82E")

```

```python
!curl https://api.openai.com/v1/fine_tuning/jobs/ftjob-Y5A5vvGYACRcuHtRf4tWLRW1 \
  -H "Authorization: Bearer ${api_key}"
```

```python
!curl https://api.openai.com/v1/files/file-iZZ17sdaLGNJpKUemqmO0LQv/content \
  -H "Authorization: Bearer ${api_key}" > file.jsonl
```

```python
# Initialize embeddings and Chroma DB
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = 'data/chroma/etsminilm'

tvdb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
```

```python
question = "major oxides or chemical compositions of materials including Al2O3, Fe2O3, CaO, MgO, So3, Tio2, Mno, K2o, Na2O, LOI, loss of ignition, ignition loss"
docs = tvdb.similarity_search(
    question,
    k=100000)
#del docs[:70000]
```

```python
def save_responses(responses):
    # Implement the logic to save 'responses'
    # For example, saving to a file:
    filename = f'responses_some.csv'
    # Convert responses to DataFrame and save as CSV
    pd.DataFrame(responses, columns=['Response', 'Caption', 'DOI']).to_csv(filename, index=False)
```

```python

```

```python

from openai import OpenAI
client = OpenAI(api_key=api_key)


```

```python
error_tables = [] 
responses = []
```

```python

for i, doc in enumerate(tqdm(docs)):

    try: 
        Table = doc.page_content  # The caption of the table
        caption = Table.split('/n')[0]
        doi = doc.metadata['doi']  # The DOI of the table

        AA = json.dumps(f"""CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
        Handling Missing Data: Use 'null' for any missing data in these columns.
        Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.

        Unstructured HTML table: {Table}""")

        # Check if this caption and DOI already exist in the existing DataFrame
        response = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal::8faiLnzJ",messages=[
        {"role": "system", "content": "You are an assistant trained to parse unstructured HTML tables into CSV format."},
        {"role": "user", "content": f"""CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
        Handling Missing Data: Use 'null' for any missing data in these columns.
        Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.

        Unstructured HTML table: {Table}"""}],
        temperature=0.0,
        max_tokens=800,
        top_p=1,)

        RR=response.choices[0].message.content
        responses.append((RR, caption, doi))

        # Check if it's time to save the responses
        if (i + 1) % 300 == 0:
            save_responses(responses)


    except Exception as e:
        error_tables.append((Table, e))  # Append the table and the error for later review
```

```python
keywords = {"al2o3", "sio2", "fe2o3", "cao", "mgo", "so3", "tio2", "mno", "k2o", "na2o", "loi", "loss of ignition", "ignition loss"}

for i, doc in enumerate(tqdm(docs)):
    if any(keyword in str(doc).lower() for keyword in keywords):
        try: 
            Table = doc.page_content  # The caption of the table
            caption = Table.split('/n')[0]
            doi = doc.metadata['doi']  # The DOI of the table

            AA = json.dumps(f"""CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
            Handling Missing Data: Use 'null' for any missing data in these columns.
            Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.

            Unstructured HTML table: {Table}""")

            # Check if this caption and DOI already exist in the existing DataFrame
            response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-1106:personal::8faiLnzJ",messages=[
            {"role": "system", "content": "You are an assistant trained to parse unstructured HTML tables into CSV format."},
            {"role": "user", "content": f"""CSV Format Requirements: Ensure the CSV includes columns for Material Name, SiO2, Al2O3, Fe2O3, CaO, MgO, SO3, TiO2, MnO, K2O, Na2O, LOI.
            Handling Missing Data: Use 'null' for any missing data in these columns.
            Non-relevant Data: If the table's content is not related to the specified chemicals, respond with 'NR' only.

            Unstructured HTML table: {Table}"""}],
            temperature=0.0,
            max_tokens=800,
            top_p=1,)

            RR=response.choices[0].message.content
            responses.append((RR, caption, doi))

            # Check if it's time to save the responses
            if (i + 1) % 300 == 0:
                save_responses(responses)


        except Exception as e:
            error_tables.append((Table, e))  # Append the table and the error for later review
```

```python
save_responses(responses)
```

```python
import pandas as pd
import io

def is_header(row, num_columns):
    """
    Check if a row can be considered as a header based on its format.
    """
    columns = row.split(",")
    if len(columns) != num_columns and column[0]=='Material Name':
        return False
    
def parse_response_to_df(response, expected_columns, caption, doi):
    # Clean the response string
    cleaned_response = str(response).replace("null","NaN")
    cleaned_response = ''.join(map(str, cleaned_response))
    cleaned_response = cleaned_response.strip('[').strip(']').strip('""').strip('''''').strip('\'"').replace('\\n', '\n').replace('\\', '')
    # Convert expected columns to lower case
    expected_columns_lower = [col.lower() for col in expected_columns]
    
    if cleaned_response == ['NR']: return pd.DataFrame(columns=expected_columns) , []
    
    # Process each row
    conf_tables = []
    
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
                new_df[exp_col] = pd.NA

        return new_df, conf_tables

    except Exception as e:
        print(e)
        print(cleaned_response)

        return pd.DataFrame(columns=expected_columns_lower), conf_tables

```

```python



# Expected columns
expected_columns = ['Material Name', 'SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'TiO2', 'MnO', 'K2O', 'Na2O', 'LOI']

# Create a DataFrame for each response
dfs = []
conf_tables = []
for response_data, caption, doi in responses:
    df , conf_table = parse_response_to_df(response_data, expected_columns, caption, doi)

    # Add 'caption' and 'doi' columns to the DataFrame
    df['caption'] = caption
    df['doi'] = doi
    
    dfs.append(df)
    conf_tables.append(conf_table)

# Concatenate all DataFrames into one
final_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=expected_columns + ['caption', 'doi'])
```

```python
# Save the DataFrame as a CSV file
final_df.to_csv('all_materials_extracted_fgpt.csv', index=False)

```
