import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_laptop_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())


plt.figure(figsize=(6,4))
sns.histplot(df['Price'], kde=True)
plt.title("Price Distribution")
plt.show()


plt.figure(figsize=(6,4))
sns.scatterplot(x=df['Ram'], y=df['Price'])
plt.title("RAM vs Price")
plt.show()


plt.figure(figsize=(6,4))
sns.scatterplot(x=df['Weight'], y=df['Price'])
plt.title("Weight vs Price")
plt.show()


numeric_df = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True)
plt.title("Correlation Heatmap")
plt.show()
