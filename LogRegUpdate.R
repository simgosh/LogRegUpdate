# Load necessary libraries
library(ISLR2)  # Library containing the Smarket dataset
library(ggplot2) # Library for visualization

# Get general information about the dataset
names(Smarket)    # List the column names in the dataset
dim(Smarket)      # Show the dimensions (number of rows and columns) of the dataset
head(Smarket)     # Display the first few rows of the dataset
summary(Smarket)  # Show summary statistics of the variables in the dataset

# Compute the correlation matrix, excluding the "Direction" column since it is categorical
cor(Smarket[, -which(names(Smarket) == "Direction")])  

# Define training data as the years before 2005
train <- Smarket$Year < 2005  

# Define test data as the year 2005
Smarket.2005 <- Smarket[!train, ]  

# Select independent variables and the dependent variable for training data
X_train <- Smarket[train, c("Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume")] # Independent variables
y_train <- Smarket$Direction[train]  # Dependent variable (whether the market went up or down)

# Create a logistic regression model
glm.fits <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume, 
                data = Smarket, 
                family = binomial,  # Using binomial logistic regression since the dependent variable is binary
                subset = train)     # Train the model using the training data

# Display model summary statistics
summary(glm.fits)  # Includes coefficients, z-statistics, p-values, etc.

# Extract model coefficients and p-values
coef(glm.fits)  # Display model coefficients
summary(glm.fits)$coef  # Show detailed coefficient summary
summary(glm.fits)$coef[, 4]  # List p-values of the model variables

# Make predictions on the test dataset
glm.probs <- predict(glm.fits, Smarket.2005, type = "response")  # Predict probabilities for the test set

# Display the first 10 predicted probabilities
glm.probs[1:10]

# Initialize all predictions as "Down" and classify using a 0.5 threshold
glm.pred <- rep("Down", nrow(Smarket.2005))  # Assign "Down" to all observations initially
glm.pred[glm.probs > 0.5] <- "Up"  # Change predictions to "Up" for probabilities greater than 0.5

# Compare model predictions with actual values
table(glm.pred, Smarket.2005$Direction)  

# Calculate model accuracy
accuracy <- mean(glm.pred == Smarket.2005$Direction)  
print(paste("Accuracy:", accuracy))  # Print accuracy

# Create a simpler model using only Lag1 and Lag2
glm.fits <- glm(Direction ~ Lag1 + Lag2, data = Smarket, 
                family = binomial, subset = train)  
 
# Make predictions using the new model on the test dataset
glm.probs <- predict(glm.fits, Smarket.2005, type = "response")  

# Again, initialize all predictions as "Down" and use a 0.5 threshold
glm.pred <- rep("Down", nrow(Smarket.2005))  
glm.pred[glm.probs > 0.5] <- "Up"  

# Compare new model predictions with actual values
table(glm.pred, Smarket.2005$Direction)  

# Calculate the accuracy of the new model
mean(glm.pred == Smarket.2005$Direction)   # Accuracy

# Compute performance metrics
TP <- conf_matrix["Up", "Up"]   # True Positive
TN <- conf_matrix["Down", "Down"] # True Negative
FP <- conf_matrix["Up", "Down"]   # False Positive
FN <- conf_matrix["Down", "Up"]   # False Negative

# Precision
precision <- TP / (TP + FP)
print(paste("Precision:", precision))

# Recall
recall <- TP / (TP + FN)
print(paste("Recall:", recall))

# Specificity
specificity <- TN / (TN + FP)
print(paste("Specificity:", specificity))

# F1 Score
F1_score <- 2 * (precision * recall) / (precision + recall)
print(paste("F1 Score:", F1_score))
