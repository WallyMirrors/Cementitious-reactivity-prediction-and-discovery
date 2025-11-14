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

filename = 'data/natural_rocks.csv'
# Load your data into a DataFrame, assuming it's a CSV for this example
df = pd.read_csv(filename, encoding='iso-8859-1')


df.head()


```

```python
desired_columns=['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'TiO2', 'MnO', 'K2O',
       'Na2O', 'LOI', 'S gravity', 'Amorphous content', 'd50', 'Ca(OH)2',
       'CaCO3', 'Water', 'K2SO4', 'KOH', 'curing temp', 'age',]
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
new_df['S gravity']=df['density_model']/1000 
new_df['d50']=10

```

```python

new_df['Ca(OH)2']=3
new_df['CaCO3']=0
new_df['Water']=3.6
new_df['K2SO4']=0
new_df['KOH']=0.151
new_df['curing temp']=50
new_df['age']=10

```

```python
new_df = new_df.clip(lower=0)
```

```python
new_df
```

```python
for column in new_df.columns:
    new_df[column] = pd.to_numeric(new_df[column], errors='coerce')
```

```python
input_mask = new_df.notnull().astype(int).values

```

```python
input_mask[0]
```

```python
df_r3 = pd.read_csv('R3.csv')
y = df_r3[['heat release', 'CH consumption', 'Bound water']]
df_r3.head()

```

```python

df_r3 = df_r3.drop(columns=['Material name','heat release','CH consumption', 'Bound water'])

for column in df_r3.columns:
    df_r3[column] = pd.to_numeric(df_r3[column], errors='coerce')
    
new_df = pd.concat([new_df, df_r3], ignore_index=True)
new_df = new_df.clip(lower=0)
```

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

    # Fill in the missing values in the original dataframe
    data.loc[data[target_column].isna(), target_column] = predicted_values

    return data

# Example usage:
# Assuming 'new_df' is your DataFrame which contains missing values in several columns
columns = new_df.columns.tolist()


# Impute each column
for column in columns:
    if new_df[column].isna().any():
        imputed_df = lightgbm_impute(new_df, column, [col for col in columns if col != column])


# Print the imputed DataFrame to verify
print(imputed_df)
```

```python
number_of_rows_in_df = len(df)

# Remove the last rows corresponding to the length of df
imputed_df = imputed_df.iloc[:number_of_rows_in_df]

```

```python
imputed_df
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

final_model = tf.keras.models.load_model('90andupnewonesch.keras', custom_objects={'custom_nan_mse': custom_nan_mse,'custom_nan_nrmse': custom_nan_nrmse})

```

```python
import pickle


mean_heat_release = y['heat release'].mean()
std_heat_release = y['heat release'].std()

mean_ch_consumption = y['CH consumption'].mean()
std_ch_consumption = y['CH consumption'].std()

mean_bound_water = y['Bound water'].mean()
std_bound_water = y['Bound water'].std()
```

```python


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
heat_release_arr = np.array(predictions_unstd[0])
ch_consumption_arr = np.array(predictions_unstd[1])
bound_water_arr = np.array(predictions_unstd[2])

```
