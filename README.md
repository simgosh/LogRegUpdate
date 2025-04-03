# SupervisedLearning
-One of the supervised learning method Logistic Regression model used, and evaluated in details.

# Summary of Code:
~~ Examining the data:
-First of all, SMarket (Stock Market Dataset) was loaded, then examining dimension of data, variable names, and summary of statistical calculations. 
- "Correlation Matrix" was calculated  without "Direction" variable, because direction variable is "categorical".

~~ Train and Test Data:
- Train data : Before 2005 years
- Test data: Only 2005 years.

~~ Creating model of Logistic Regression:
- In the first model, using the variables of "Lag1, Lag2, Lag3, Lag4, Lag5 and Volume" and estimated of "Direction" variable. (with Binomial Logistic Reg.)
- Evaluation of models coefficients and p-value.

~~ Making prediction and evaluating model performance:
- Model is making probabilty estimation on test data.
- Threshold value is making classification by 0.5.
- Model's accuracy rate is calculated.

~~ Simple Model (without overfitting risk):
- Creating a new model, using just "Lag1 and Lag2" variables.
- New model's predictions are making test and accuracy value was calculated again.

~~ Model Performance Evulation:
- Confusion matrix, precision, accuracy, recall, f1 score and ROC were calculated, and interpreted.

