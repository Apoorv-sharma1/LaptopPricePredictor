import numpy as np
import pandas as pd

df = pd.read_csv('laptop_data.csv')

df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.head(5))

df['Ram'] = df['Ram'].str.replace('GB', '')
df['Weight'] = df['Weight'].str.replace('kg', '')

df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

print("Missing Values:\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

df.to_csv('cleaned_laptop_data.csv', index=False)