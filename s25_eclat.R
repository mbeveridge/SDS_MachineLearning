# Eclat

# Data Preprocessing [§25 Lect167: "Eclat in R" ...This § same as for Apriori lecture]
# install.packages('arules') [Note: Anaconda doesn't have the `arules` package]
library(arules)
dataset = read.csv('data/s25_Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('data/s25_Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset [§25 Lect167: "Eclat in R"]
rules = eclat(data = dataset, parameter = list(support = 0.003, minlen = 2))

# Visualising the results [§25 Lect167: "Eclat in R"]
inspect(sort(rules, by = 'support')[1:10])