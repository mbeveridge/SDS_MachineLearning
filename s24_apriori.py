# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing [§24 Lect162: "Apriori in Python - Step 1"]
dataset = pd.read_csv('data/s24_Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset [§24 Lect163: "Apriori in Python - Step 2"]
from apyori import apriori #  [Note: Anaconda doesn't have `apyori`; portable `apyori.py` file is in repo]
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results [§24 Lect164: "Apriori in Python - Step 3"]
results = list(rules)