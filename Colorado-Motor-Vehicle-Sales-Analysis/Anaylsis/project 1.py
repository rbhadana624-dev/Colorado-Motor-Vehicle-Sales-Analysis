"""This code loads a dataset of motor vehicle sales in Colorado, 
checks for missing values, creates a proper time index, and 
visualizes the total sales over time"""

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\\Users\\rahul\\Downloads\\colorado_motor_vehicle_sales.csv")
df.head()

# Check for missing values
print(df.shape)
print(df.info())
print(df.isnull().sum())

# creating proper time index
df['period'] = df['year'].astype(str) + 'Q' + df['quarter'].astype(str)

# Convert 'year' and 'quarter' to datetime
df['month'] = (df['quarter'] - 1) * 3 + 1
df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
df.set_index('date', inplace=True)

# EDA and visualization
import matplotlib.pyplot as plt

total_sales = df.groupby(['year', 'quarter'])['sales'].sum().reset_index()

plt.figure(figsize=(10,6))
plt.plot(total_sales['sales'])
plt.title("Total Motor Vehicle Sales Over Time")
plt.show()

#Sales Distribution by Quarter
import seaborn as sns

sns.boxplot(x='quarter', y='sales', data=df)
plt.title("Sales Distribution by Quarter")
plt.show()

#country wise sales distribution
county_sales = df.groupby('county')['sales'].sum().sort_values(ascending=False)
print(county_sales.head(10))

#Yearly Growth Rate
yearly_sales = df.groupby('year')['sales'].sum()
growth_rate = yearly_sales.pct_change() * 100
print(growth_rate)

#Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

quarterly_sales = df.groupby(df.index)['sales'].sum()
result = seasonal_decompose(quarterly_sales, model='multiplicative', period=4)
result.plot()
plt.show()

"""Predictive modeling using Random forest"""
df_ml = df.copy()
df_ml['county'] = df_ml['county'].astype('category').cat.codes

X = df_ml[['year', 'quarter', 'county']]
y = df_ml['sales']

#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# train model and evaluate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)