---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: soroush4
    language: python
    name: soroush4
---

```python
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

from sklearn.impute import KNNImputer

```

```python
import pandas as pd


df = pd.read_csv('Final_ext_compositions_w_subcategories.csv')
df
```

```python
desired_columns=['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'TiO2', 'MnO', 'K2O',
       'Na2O', 'LOI', 'S gravity', 'Amorphous content']
```

```python
# Create a new DataFrame with the desired columns
new_df = pd.DataFrame()

for column in desired_columns:
    if column in df.columns:
        # Add the existing column to the new DataFrame
        new_df[column] = df[column]
    else:
        # Case insensitive check and add if exists
        matches = [col for col in df.columns if col.lower() == column.lower()]
        if matches:
            new_df[column] = df[matches[0]]
        else:
            # Create a column with NaN values if it doesn't exist
            new_df[column] = np.nan

# Display the head of the new DataFrame to verify
print(new_df.head())
```

```python
new_df['d50']=10
new_df['Ca(OH)2']=3
new_df['CaCO3']=0
new_df['Water']=3.6
new_df['K2SO4']=0
new_df['KOH']=0.151
new_df['curing temp']=50
new_df['age']=10
new_df
```

```python
new_df['d50'][df['Category']=='Cements']=14.11
new_df['d50'][df['Category']=='Clinkers']=20
new_df['d50'][df['Category']=='Fly ashes']=8.88
new_df['d50'][df['Category']=='Slags']=6.15
new_df['d50'][df['Category']=='Glasses']=24.99

```

```python
new_df['d50'][df['Category']=='Silica fume']=0.2
new_df['S gravity'][df['Category']=='Silica fume']=2.16
new_df['LOI'][df['Category']=='Silica fume']=6.04

```

```python
input_mask = new_df.notnull().astype(int).values
input_mask
```

```python
df_r3 = pd.read_csv('R3.csv')
y = df_r3[['heat release', 'CH consumption', 'Bound water']]

df_r3 = df_r3.drop(columns=['Material name','heat release','CH consumption', 'Bound water'])

for column in df_r3.columns:
    df_r3[column] = pd.to_numeric(df_r3[column], errors='coerce')
    
Updated_compositions = pd.concat([new_df, df_r3], ignore_index=True)
Updated_compositions=Updated_compositions[desired_columns]
```

```python
Updated_compositions = pd.concat([new_df, df_r3], ignore_index=True)

```

```python
# Save the 'Category' column into a new variable
categories = df['Category']
matnames = df['Sub_category']

# Drop the 'Category' column
Updated_compositions.head()
```

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Ensure all columns are numeric
for column in Updated_compositions.columns:
    Updated_compositions[column] = pd.to_numeric(Updated_compositions[column], errors='coerce')

def lightgbm_impute(data, target_column, features):
    # Split data into training (known target values) and testing (unknown target values)
    known = data[data[target_column].notna()]
    unknown = data[data[target_column].isna()]

    # If no missing values are present, nothing needs to be done
    if unknown.shape[0] == 0:
        return data

    # Prepare input features and target
    X_train = known[features].values
    y_train = known[target_column].values
    X_test = unknown[features].values

    # Determine if the target column is categorical
    if issubclass(data[target_column].dtype.type, np.integer):
        model = lgb.LGBMClassifier()  # Use classifier for categorical data
    else:
        model = lgb.LGBMRegressor()  # Use regressor for numerical data

    # Train the model
    model.fit(X_train, y_train)

    # Predict the missing values
    predicted_values = model.predict(X_test)

    # Fill in the missing values in the original dataframe without altering the known values
    data.loc[data[target_column].isna(), target_column] = predicted_values

    return data

# Example usage:
# Assuming 'Updated_compositions' is your DataFrame which contains missing values in several columns
columns = Updated_compositions.columns.tolist()

# Impute each column
for column in columns:
    if Updated_compositions[column].isna().any():
        Updated_compositions = lightgbm_impute(Updated_compositions, column, [col for col in columns if col != column])

# Print the imputed DataFrame to verify
print(Updated_compositions)

```

```python
Updated_compositions[Updated_compositions < 0] = 0

```

```python
number_of_rows_in_df = len(df)

imputed_df = Updated_compositions.iloc[:number_of_rows_in_df]
input_mask = input_mask[:number_of_rows_in_df]

```

```python
def custom_nan_mse(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, dtype=tf.float32)
    
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    squared_difference = tf.square(y_true - y_pred) * mask
    
    # Add a small constant (epsilon) to avoid division by zero
    epsilon = 1e-7
    return tf.reduce_sum(squared_difference) / (tf.reduce_sum(mask) + epsilon)

def custom_nan_nrmse(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    mask = tf.cast(mask, dtype=tf.float32)
    
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    squared_difference = tf.square(y_true - y_pred) * mask
    mse = tf.reduce_sum(squared_difference) / tf.reduce_sum(mask)
    rmse = tf.sqrt(mse)
    
    # Mask y_true values, replacing masked values with infinity and negative infinity
    # to ensure they don't affect the min and max calculations
    y_true_masked_min = tf.where(mask > 0, y_true, tf.fill(tf.shape(y_true), tf.float32.max))
    y_true_masked_max = tf.where(mask > 0, y_true, tf.fill(tf.shape(y_true), tf.float32.min))
    
    # Compute the min and max of the non-NaN y_true values
    y_true_min = tf.reduce_min(y_true_masked_min)
    y_true_max = tf.reduce_max(y_true_masked_max)
    
    y_range = y_true_max - y_true_min
    epsilon = 1e-7
    nrmse = rmse / (y_range + epsilon)
    
    return nrmse



```

```python
import tensorflow as tf

final_model = tf.keras.models.load_model('90andupnewones.keras', custom_objects={'custom_nan_mse': custom_nan_mse,'custom_nan_nrmse': custom_nan_nrmse})

```

```python
mean_heat_release = y['heat release'].mean()
std_heat_release = y['heat release'].std()

mean_ch_consumption = y['CH consumption'].mean()
std_ch_consumption = y['CH consumption'].std()

mean_bound_water = y['Bound water'].mean()
std_bound_water = y['Bound water'].std()
```

```python
import pickle 
with open('df_imputed.pkl', 'rb') as file:
    df_imputed = pickle.load(file)

scaler = StandardScaler()

X_imputed = df_imputed.drop(columns=['heat release', 'CH consumption', 'Bound water'])
X_scaled = scaler.fit_transform(X_imputed)
```

```python
def unstandardize(values, mean, std):
    """Convert standardized values back to the original scale."""
    return values * std + mean
```

```python
len(X_scaled[0])
```

```python
X_scaled = scaler.transform(imputed_df)

predictions = final_model.predict([X_scaled,input_mask] )

print(predictions)
```

```python
# Unstandardize predictions
predictions_unstd = [
    unstandardize(predictions[0].flatten(), mean_heat_release, std_heat_release),
    unstandardize(predictions[1].flatten(), mean_ch_consumption, std_ch_consumption),
    unstandardize(predictions[2].flatten(), mean_bound_water, std_bound_water)
]
```

```python
ch_consumption_arr = np.array(predictions_unstd[1])
bound_water_arr = np.array(predictions_unstd[2])
heat_release_arr = np.array(predictions_unstd[0])


# Find indices where the value is greater than 380
indices_heat = np.where(heat_release_arr > 380)[0]
# Find indices where bound_water_arr > 10
indices_water = np.where(bound_water_arr > 10)[0]

# Find common indices satisfying both conditions
# First, intersect the first two arrays
common_indices = np.intersect1d(indices_heat, indices_water)


df['Heat release']=heat_release_arr
df['CH consumption']=ch_consumption_arr
df['Bound water']=bound_water_arr

# Filter the dataframe using these indices
filtered_df = df.iloc[common_indices]

# Display the filtered dataframe
print(filtered_df)
```
