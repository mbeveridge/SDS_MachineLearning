# Polynomial Regression

# Importing the dataset
dataset = read.csv('data/s6_Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# Not done in this case, because only ten observations are available
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Linear Regression to the dataset [§6 Lect62: "Polynomial Regression - R pt2" ...for comparison]
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# Fitting Polynomial Regression to the dataset [§6 Lect62: "Polynomial Regression - R pt2"]
dataset$Level2 = dataset$Level^2 # Add a column (called "Level2") to the dataframe, which is Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4 # Video eventually says (Lect63 17'30-18'40) to include degree = 4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)

# Visualising the Linear Regression results [§6 Lect63: "Polynomial Regression - R pt3" ...for comparison]
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Polynomial Regression results [§6 Lect63: "Polynomial Regression - R pt3"]
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model results (add x_grid ...if we want higher resolution and smoother curve)
# [§6 Lect63: "Polynomial Regression - R pt3" at 19'30 says this will be in the file (but not in video)]
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg,
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid^2,
                                                             Level3 = x_grid^3,
                                                             Level4 = x_grid^4))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# Predicting a new result with Linear Regression [§6 Lect64: "Polynomial Regression - R pt4" ...for comparison]
y_pred = predict(lin_reg, data.frame(Level = 6.5)) # This is now a single prediction, not a vector

# Predicting a new result with Polynomial Regression [§6 Lect64: "Polynomial Regression - R pt4"]
y_pred = predict(poly_reg, data.frame(Level = 6.5,
                             Level2 = 6.5^2,
                             Level3 = 6.5^3,
                             Level4 = 6.5^4))
