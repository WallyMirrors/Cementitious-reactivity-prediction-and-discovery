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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time

```

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

# Load the dataset
df = pd.read_csv('R3.csv')
mn= df['Material name']
df = df.drop(columns=['Material name'])

# Replace empty strings with NaN
df = df.replace(' ', np.nan)

# Convert all columns to numeric, coercing errors to NaN
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Separate the features and target variables if needed
# Here, assuming all columns except 'Material name' are features
df.head()
```

```python

# Separate features and targets
X = df.drop(columns=['heat release', 'CH consumption', 'Bound water'])
y = df[['heat release', 'CH consumption', 'Bound water']]

```

```python
X
```

```python
input_mask = X.notnull().astype(int).values

```

```python
import pickle

# Load the imputed X matrix from the pickle file
with open('df_imputed.pkl', 'rb') as file:
    df_imputed = pickle.load(file)
# Normalize features
scaler = StandardScaler()

X_imputed = df_imputed.drop(columns=['heat release', 'CH consumption', 'Bound water'])
X_scaled = scaler.fit_transform(X_imputed)



```

```python
X_imputed
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
# Scatter plot for training data
# Set the aesthetic style of the plots
plt.style.use('seaborn-white')  # Use a clean white style
sns.set_style("ticks")  # Use ticks style for a more refined look
plt.rcParams['font.family'] = 'sans-serif'  # Use a sans-serif font for a modern look
plt.rcParams['font.size'] = 12  # Adjust font size for readability

fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the size for a square aspect ratio

sns.scatterplot(y='d50', x='heat release',  data=df, s=20)
ax.set_xlabel('Actual heat release (J/g)')
ax.set_ylabel('Median size (µm)')


plt.ylim(0,100)
plt.ylim(1,60)

ax.legend(frameon=False,prop={'size': 12})  # Remove the legend frame for a cleaner look
sns.despine()  # Remove the top and right spines
```

```python

mean_heat_release = y['heat release'].mean()
std_heat_release = y['heat release'].std()

mean_ch_consumption = y['CH consumption'].mean()
std_ch_consumption = y['CH consumption'].std()

mean_bound_water = y['Bound water'].mean()
std_bound_water = y['Bound water'].std()

# Training set
y['heat release'] = (y['heat release'] - mean_heat_release) / std_heat_release
y['CH consumption'] = (y['CH consumption'] - mean_ch_consumption) / std_ch_consumption
y['Bound water'] = (y['Bound water'] - mean_bound_water) / std_bound_water

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
def unstandardize(values, mean, std):
    """Convert standardized values back to the original scale."""
    return values * std + mean
```

```python
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

# Assuming custom_nan_mse, custom_nan_nrmse, weight_heat_release, weight_ch_consumption, and weight_bound_water are defined elsewhere

# Define input layers for features and their masks
input_features = Input(shape=(X_scaled.shape[1],), name='Input_Features')
input_masks = Input(shape=(X_scaled.shape[1],), name='Input_Masks')

# Combine the inputs and masks by concatenation
combined_inputs = concatenate([input_features, input_masks])

# Define the rest of the model
hidden1 = Dense(256, activation='relu')(combined_inputs)
hidden2 = Dropout(0.1)(hidden1)

hidden3 = Dense(256, activation='relu')(hidden2)
hidden4 = Dense(16, activation='relu')(hidden3)

# Define the output layers with names
Heat_Release_Output = Dense(1, name='Heat_Release_Output')(hidden4)
CH_Consumption_Output = Dense(1, name='CH_Consumption_Output')(hidden4)
Bound_Water_Output = Dense(1, name='Bound_Water_Output')(hidden4)

# Create the model
model = Model(inputs=[input_features, input_masks], 
              outputs=[Heat_Release_Output, CH_Consumption_Output, Bound_Water_Output])

# Assuming these are the counts of non-NaN data points for each target
count_heat_release = np.count_nonzero(~np.isnan(y_train['heat release']))
count_ch_consumption = np.count_nonzero(~np.isnan(y_train['CH consumption']))
count_bound_water = np.count_nonzero(~np.isnan(y_train['Bound water']))

# Calculate the total count for normalization
total_count = count_heat_release + count_ch_consumption + count_bound_water

global weight_heat_release, weight_ch_consumption, weight_bound_water

# Calculate loss weights inversely proportional to the number of data points
weight_heat_release = (total_count - count_heat_release) / total_count
weight_ch_consumption = (total_count - count_ch_consumption) / total_count
weight_bound_water = (total_count - count_bound_water) / total_count
# Compile the model with specified loss weights
model.compile(optimizer='adam',
              loss={'Heat_Release_Output': custom_nan_mse,
                    'CH_Consumption_Output': custom_nan_mse,
                    'Bound_Water_Output': custom_nan_mse},
              loss_weights={'Heat_Release_Output': weight_heat_release,
                            'CH_Consumption_Output': weight_ch_consumption*5,
                            'Bound_Water_Output': weight_bound_water},
              metrics={'Heat_Release_Output': [custom_nan_mse, custom_nan_nrmse],
                       'CH_Consumption_Output': [custom_nan_mse, custom_nan_nrmse],
                       'Bound_Water_Output': [custom_nan_mse, custom_nan_nrmse]})

```

```python
weight_bound_water
```

```python
# Split the data and masks into training and testing sets

# Create masks for input features (1 for present values, 0 for missing values)

from tensorflow.keras.callbacks import EarlyStopping

# Split the data and masks into training and testing sets
# Split the data and masks into training and testing sets

num_bins = 5
bins_heat_release = np.linspace(df_imputed['heat release'].min(), df_imputed['heat release'].max(), num_bins)
bins_ch_consumption = np.linspace(df_imputed['CH consumption'].min(), df_imputed['CH consumption'].max(), num_bins)
bins_bound_water = np.linspace(df_imputed['Bound water'].min(), df_imputed['Bound water'].max(), num_bins)

# Initialize yy as a dictionary
yy = {}

# Digitize (bin) each target
yy['heat_release_binned'] = np.digitize(y['heat release'], bins_heat_release)
yy['ch_consumption_binned'] = np.digitize(y['CH consumption'], bins_ch_consumption)
yy['bound_water_binned'] = np.digitize(y['Bound water'], bins_bound_water)

# Calculate the maximum number of bins to set the radix
max_bins = max(bins_heat_release.size, bins_ch_consumption.size, bins_bound_water.size)

# Create a combined stratification key as a single integer
stratify_key = yy['heat_release_binned'] + \
               (yy['ch_consumption_binned'] * max_bins) + \
               (yy['bound_water_binned'] * (max_bins ** 2))
# Create masks for input features (1 for present values, 0 for missing values)
input_mask = X.notnull().astype(int).values
X_train, X_test, mask_train, mask_test, y_train, y_test = train_test_split(
    X_scaled, input_mask, y, test_size=0.2, random_state=42, stratify =stratify_key)
from tensorflow.keras.callbacks import EarlyStopping

# Train the model
early_stopping = EarlyStopping(
    monitor='val_CH_Consumption_Output_custom_nan_nrmse',  # Metric to monitor
    patience=200,         # Number of epochs to wait after min has been hit
    mode='min',          # Stop when the monitored quantity has stopped decreasing
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

model.fit([X_train, mask_train], 
                       {'Heat_Release_Output': y_train['heat release'],
                        'CH_Consumption_Output': y_train['CH consumption'],
                        'Bound_Water_Output': y_train['Bound water']},
                       epochs=10000, 
                       batch_size=50, 
                       validation_split=0.15,
                       callbacks=[early_stopping])  # Set verbose to 0 to reduce training output
    
```

```python


# Unstandardize y_test values
y_test_unstd = {
    'heat release': unstandardize(y_test['heat release'], mean_heat_release, std_heat_release),
    'CH consumption': unstandardize(y_test['CH consumption'], mean_ch_consumption, std_ch_consumption),
    'Bound water': unstandardize(y_test['Bound water'], mean_bound_water, std_bound_water)
}

y_train_unstd = {
    'heat release': unstandardize(y_train['heat release'], mean_heat_release, std_heat_release),
    'CH consumption': unstandardize(y_train['CH consumption'], mean_ch_consumption, std_ch_consumption),
    'Bound water': unstandardize(y_train['Bound water'], mean_bound_water, std_bound_water)
}

predictions = model.predict([X_test,mask_test] )

# Unstandardize predictions
predictions_test_unstd = [
    unstandardize(predictions[0].flatten(), mean_heat_release, std_heat_release),
    unstandardize(predictions[1].flatten(), mean_ch_consumption, std_ch_consumption),
    unstandardize(predictions[2].flatten(), mean_bound_water, std_bound_water)
]

predictions = model.predict([X_train,mask_train] )

# Unstandardize predictions
predictions_train_unstd = [
    unstandardize(predictions[0].flatten(), mean_heat_release, std_heat_release),
    unstandardize(predictions[1].flatten(), mean_ch_consumption, std_ch_consumption),
    unstandardize(predictions[2].flatten(), mean_bound_water, std_bound_water)
]

```

```python
len(X_test)
```

```python
import tensorflow as tf

model = tf.keras.models.load_model('90andupnewones.keras', custom_objects={'custom_nan_mse': custom_nan_mse,'custom_nan_nrmse': custom_nan_nrmse})

```

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Prepare DataFrames for plotting
df_plot_test = pd.DataFrame({
    'heat release': y_test_unstd['heat release'],
    'heat_release_prediction': predictions_test_unstd[0]
})

df_plot_train = pd.DataFrame({
    'heat release': y_train_unstd['heat release'],
    'heat_release_prediction': predictions_train_unstd[0]
})

# Filter out rows with NaN values
df_filtered_test = df_plot_test.dropna()
df_filtered_train = df_plot_train.dropna()

# Set the aesthetic style of the plots
plt.style.use('seaborn-white')  # Use a clean white style
sns.set_style("ticks")  # Use ticks style for a more refined look
plt.rcParams['font.family'] = 'sans-serif'  # Use a sans-serif font for a modern look
plt.rcParams['font.size'] = 12  # Adjust font size for readability

# Create a combined plot for both training and testing datasets
fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the size for a square aspect ratio

# Calculate and display R² scores
r2_test = r2_score(df_filtered_test['heat release'], df_filtered_test['heat_release_prediction'])
r2_train = r2_score(df_filtered_train['heat release'], df_filtered_train['heat_release_prediction'])


# Scatter plot for training data
sns.regplot(x='heat release', y='heat_release_prediction', data=df_filtered_train,
            scatter_kws={'alpha': 0.5, 'color': '#DF602A'}, line_kws={'color': '#DF602A', 'label': f'Train (R²: {r2_train:.2f})'},
            ax=ax, ci=0)

# Scatter plot for testing data
sns.regplot(x='heat release', y='heat_release_prediction', data=df_filtered_test,
            scatter_kws={'alpha': 0.5, 'color': '#7DCFB6'}, line_kws={'color': '#7DCFB6', 'label': f'Test (R²:{r2_test:.2f})'},
            ax=ax, ci=0)

# Add the zero error line last and use zorder to ensure it's on top
ax.plot([df_filtered_train['heat release'].min(), df_filtered_train['heat release'].max()], 
        [df_filtered_train['heat release'].min(), df_filtered_train['heat release'].max()], 
        'k--', lw=2, label='Zero error', zorder=5)

# Customize the plot
ax.set_xlabel('Actual heat release (J/g)')
ax.set_ylabel('Predicted heat release (J/g)')
ax.legend(frameon=False,prop={'size': 12})  # Remove the legend frame for a cleaner look
sns.despine()  # Remove the top and right spines

# Adjust ticks for a more refined appearance
ax.tick_params(axis='both', which='major', length=5, width=1, direction='out')

# Save the figure as a high-quality PNG file
plt.savefig('goodness_of_fit_heat_release.png', dpi=300, bbox_inches='tight', transparent=True)

plt.show()

```

```python
model.save('90andupnewonesch.keras')  # Saves to the HDF5 format

```
