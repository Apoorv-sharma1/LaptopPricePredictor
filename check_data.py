import pandas as pd
import numpy as np

df = pd.read_csv('laptop_data.csv')
print("Columns:", df.columns.tolist())
print("\nUnique Companies:", df['Company'].unique()[:5])
print("\nUnique Screen Resolutions:", df['ScreenResolution'].unique()[:5])
print("\nUnique CPU:", df['Cpu'].unique()[:5])
print("\nUnique Memory:", df['Memory'].unique()[:5])
print("\nUnique GPU:", df['Gpu'].unique()[:5])
print("\nUnique OS:", df['OpSys'].unique()[:5])
