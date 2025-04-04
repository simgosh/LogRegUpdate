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
from scipy.stats import f

#load original data
df = pd.read_csv("~/Desktop/Advertising.csv")
print(df.head())
df.drop(columns="Unnamed: 0", inplace=True)
print(df.columns)
print(df.isnull().sum())
print(df.dtypes)
print(df.value_counts().sum())

#check out correlation for all variables
corr_matrix = df.corr()
print(corr_matrix) #sales and tv have positive strong relationship.
sns.heatmap(data=corr_matrix,annot=True, fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#evulated model
X = df[["TV", "radio"]] #newspaper is not very important for model (by the model perfomance metrics)
y = df["sales"].values

print(X.shape)
print(y.shape)

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)  # (160, 1)
print(y_train.shape)  # (160,)
print(X_test.shape)   # (40, 1)
print(y_test.shape)  

#created model
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print('Coefficients:', lr.coef_)
print('Intercept:', lr.intercept_) #Y = 3.029 + 0.0448*x1 + 0.19066*x2

mse= mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}") #10.20465 
print(f"R-Squared : {r2}") #0.6766

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Sales Values')
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

#### Creating anova table ####
##### 1. Total Sum of Squares (SST)
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

if p_value<0.05:
    print(f"H0 is rejected, model is significant. : {p_value}")
else:
    print(f"H0 is not rejected, model is not significiant")

#f-table vs f-test
df1 = 2  # number of predictors
df2 = 37  # denominator degrees of freedom 

# F-kritik values for α = 0.05  
f_critical = f.ppf(1 - 0.05, df1, df2)
print(f_critical) # 3.251923

if f_value > f_critical:
    print(f"H0 is rejected, and model is significiant. {f_value} > {f_critical}")
else:
    print(f"H0 is not rejected. Model is not significiant.")

X = sm.add_constant(df[["TV", "radio"]])

# Bağımlı değişkeni (y)
y = df["sales"]

# Modeli oluştur ve fit et
model = sm.OLS(y, X)
results = model.fit()

# Sonuçları özetle
print(results.summary()) #Y=2.9211+0.0458⋅TV+0.1880⋅Radio

residuals1 = results.resid

# Residual Sum of Squares (SSE)
sse1 = np.sum(residuals1**2)

# Mean Squared Error (MSE)
n = len(y)  # veri sayısı
k = X.shape[1]  # bağımsız değişken sayısı
mse1 = sse1 / (n - k - 1)

print(f"Mean Squared Error (MSE): {mse1}")

# Summary **Modeling Method:**
#    - **OLS (statsmodels)**: OLS directly estimates the coefficients using the ordinary least squares method. The OLS model provides more detailed statistical outputs, including not only the regression coefficients but also an analysis of error terms (t-statistics, p-values, standard errors).
#    - **Linear Regression (sklearn)**: LinearRegression is a simpler model that typically uses optimization techniques (such as Gradient Descent or the Normal Equation) to estimate the
