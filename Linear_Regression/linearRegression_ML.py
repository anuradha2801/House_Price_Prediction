
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("C:/Users/91930/Documents/datasetML/Housing.csv")

# Split the dataset into training and testing sets
X_train = data["area"].values.reshape(-1, 1)[:250]
y_train = data["price"].values.reshape(-1, 1)[:250]
X_test = data["area"].values.reshape(-1, 1)[250:]
y_test = data["price"].values.reshape(-1, 1)[250:]

# Train the linear regression model on the training data
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the price of a new house with area=1000
new_house_area = 1000
predicted_price = regressor.predict([[new_house_area]])
print("Predicted price:", predicted_price)

