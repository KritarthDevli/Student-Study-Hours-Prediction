import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_excel("student.xlsx")
data.columns = data.columns.str.strip()

X = data[['hours']]
y = data['score']

model = LinearRegression()
model.fit(X, y)

predicted_score = model.predict(X)

mae = mean_absolute_error(y, predicted_score)
mse = mean_squared_error(y, predicted_score)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

new_prediction = model.predict([[7]])
print("If study for  7 hours, predicted score =", new_prediction[0])

new_hour = float(input("Enter number of study hours: "))
new_pred = model.predict([[new_hour]])
print(f"Predicted score for {new_hour} hours = {new_pred[0]:.2f}")
