# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('data/s5_50_Startups.csv')
# dataset = dataset[, 2:3]

# Encoding categorical data [§5 Lect48: "MLR - R pt1" ...@3min00]
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set [§5 Lect49: "MLR - R pt2"]
regressor = lm(formula = Profit ~ .,
               data = training_set)

# Predicting the Test set results [§5 Lect50: "MLR - R pt3"]
y_pred = predict(regressor, newdata = test_set)

# Building the optimal model using Backward Elimination [§5 Lect51: "MLR - R (Backward Elimination): HOMEWORK !"]
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)
# Optional Step: Remove State2 only (as opposed to removing State directly)
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + factor(State, exclude = 2),
#                data = dataset)
# summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)
# cont'd [§5 Lect52: "MLR - R (Backward Elimination): Homework solution"]
regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)
regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)

