import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("cleaned_laptop_data.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
accuracy = 100 - (np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("Accuracy:", accuracy)

if r2 > 0.8:
    print("Model Performance: Very Good")
elif r2 > 0.6:
    print("Model Performance: Good")
elif r2 > 0.4:
    print("Model Performance: Average")
else:
    print("Model Performance: Poor")

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (KNN Regression)")
plt.show()

plt.figure()
errors = y_test - y_pred
plt.hist(errors)
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()