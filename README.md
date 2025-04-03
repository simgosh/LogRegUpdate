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

~~ Simplified Model (without overfitting risk):
- Creating a new model, using just "Lag1 and Lag2" variables.
- New model's predictions are making test and accuracy value was calculated again.

~~ Model Performance Evulation:
- Confusion matrix, precision, accuracy, recall, f1 score and ROC were calculated, and interpreted.

# Interpreted Model Performances:
~~ Performance of First Model (include all variables):
- Lag1, Lag2, Lag3, Lag4, Lag5 and Volume were used.
- However, in the financial sector, price changes over the past days are not very significant.
- Correlation matrix, can show there is no strong relationship among variables.
- Model's accuracy is almost 50%, this result can tell us, this is close to estimation of random completely.

~~Simplified Model (Only Lag1 and Lag2):
- Although, this model involved less variables, result is similar the former one model's.
- This result tell us, other Lag variables doesn't as much as additive for model.

~~ Performance Metrics:
- Precision : A precision value of 0.58 indicates that the model is 58% accurate in its predictions of "Up". This means that the model is wrong about 42% of the time in its predictions for the "Up" class, so the margin of error is quite high.
- Recall : A recall value of 0.75 indicates that the model correctly predicted 75% of the true instances that were “Up”. This means that it correctly classified most of its observations, but some observations in the “Up” class are missed.
- Specificity : A specificity value of 0.32 indicates that the model has a low rate of correctly predicting observations that are "Down." This indicates that the model has a high error rate in its predictions of "Down," indicating that the model has a particularly difficult time predicting this class.
- F1 : A F1 score of 0.66 indicates that the model strikes a balance between precision and recall, but still needs improvement in both metrics. That is, the model is better at correctly predicting the "Up" class, but has gaps in predictions in the "Down" class.
