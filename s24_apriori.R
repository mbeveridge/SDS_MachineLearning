# Apriori

# Data Preprocessing [§24 Lect159: "Apriori in R - Step 1"]
# install.packages('arules') [Note: Anaconda doesn't have the `arules` package]
library(arules)
dataset = read.csv('data/s24_Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('data/s24_Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset [§24 Lect160: "Apriori in R - Step 2"]
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results [§24 Lect161: "Apriori in R - Step 3"]
inspect(sort(rules, by = 'lift')[1:10])