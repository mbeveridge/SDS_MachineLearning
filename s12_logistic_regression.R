# Logistic Regression

# Importing the dataset
dataset = read.csv('data/s12_Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor [not shown in a §12 video]
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set [§12 Lect92: "Logistic Regression - R pt1"]
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling [§12 Lect92: "Logistic Regression - R pt1"]
# ...Video had `[, 1:2]` not `[-3]`, but in R the latter means remove the 3rd col (leaving 1&2)
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Fitting Logistic Regression to the Training set [§12 Lect93: "Logistic Regression - R pt2"]
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)

# Predicting the Test set results [§12 Lect94: "Logistic Regression - R pt3"]
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix [§12 Lect95: "Logistic Regression - R pt4"]
cm = table(test_set[, 3], y_pred > 0.5)

# Visualising the Training set results [§12 Lect96: "Logistic Regression - R pt5"]
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

# Visualising the Test set results [§12 Lect96: "Logistic Regression - R pt5"]
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


# ggplot2 (instead, roughly) for visualising the Training set results
# ...[§12 Lect96: "Logistic Regression - R pt5" Q&A]
# [https://www.udemy.com/machinelearning/learn/v4/questions/4164774]
library(tidyverse)
set = training_set
expand.grid('Age' = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01),
            'EstimatedSalary' = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01))%>%
  mutate(prob_set=predict(classifier, type = 'response', newdata = .),
         y_grid = ifelse(prob_set > 0.5, 1, 0))%>%
  ggplot()+
  geom_point(aes(x=.$Age, y=.$EstimatedSalary, color=.$y_grid))+
  geom_point(data=training_set, aes(x=Age, y=EstimatedSalary,colour=as.numeric(Purchased)))

