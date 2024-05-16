import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('captions.csv')

# Split the data into training, validation, and test sets
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.25, random_state=42)

# Save the splits
train_data.to_csv('train_data.csv', index=False)
val_data.to_csv('val_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
