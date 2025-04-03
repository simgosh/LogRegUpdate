import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols

file_path = '/Users/sim/Desktop/Boston.csv'
df = pd.read_csv(file_path)

#checking data
print(df.head())
print(df.describe())
df.drop(columns="Unnamed: 0", inplace=True)
print(df.columns)
print(df.isnull().sum())
print(df["crim"].mean().round(2))
print(df['medv'].mean().round(3)) #dependent (response) variable 


X = df[["lstat", "age", "tax", "rm"]]
y= df["medv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

mse= mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-Squared : {r2}")

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek ve Tahmin Edilen Ev Fiyatları')
plt.show()

#residuals
residuals = y_test - y_pred
print(residuals)

plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

###### 1. Total Sum of Squares (SST)
y_mean = y_test.mean()
sst = np.sum((y_test - y_mean) ** 2)

# 2. Regression Sum of Squares (SSR)
ssr = np.sum((y_pred - y_mean) ** 2)

# 3. Error Sum of Squares (SSE)
sse = np.sum((y_test - y_pred) ** 2)

# 4. Degrees of freedom
df_regression = X_train.shape[1] # number of predictors
df_error = len(y_test) - df_regression - 1
df_total = len(y_test) - 1

#5. Mean Squares
msr = ssr / df_regression

# 6. F-statistic
f_value = msr / mse

#pvalue 
p_value = 1 - stats.f.cdf(f_value, df_regression, df_error)

# Print ANOVA Table
print("\nANOVA Table")
print("---------------------------------------------------------")
print(f"Regression, SSR: {ssr}, df: {df_regression}, MSR: {msr}")
print(f"Error, SSE: {sse}, df: {df_error}, MSE: {mse}")
print(f"Total, SST: {sst}, df: {df_total}")
print(f"F-Statistic: {f_value}")
print(f"P-value: {p_value}")


###
# make estimation for new variables
new_data = pd.DataFrame({
    'lstat': [12, 15, 20],
    'age': [70, 60, 80],
    'tax': [300, 200, 400],
    'rm': [6, 6.5, 5.8]
})

new_predictions = model.predict(new_data)
print(f"Yeni tahminler: {new_predictions}")
