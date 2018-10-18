# Natural Language Processing

# Importing the dataset [§29 Lect202: "NLP in R - pt1"]
dataset_original = read.delim('data/s29_Restaurant_Reviews.tsv', quote = '', stringsAsFactors = FALSE)

# Cleaning the texts [§29 Lect203: "NLP in R - pt2"] & [§29 Lect204: "NLP in R - pt3"]
# & [§29 Lect205: "NLP in R - pt4"] & [§29 Lect206: "NLP in R - pt5"] & [§29 Lect207: "NLP in R - pt6"]
# & [§29 Lect208: "NLP in R - pt7"] & [§29 Lect209: "NLP in R - pt8"]
# install.packages('tm')
# install.packages('SnowballC')
library(tm)                                                   # "NLP in R - pt2"
library(SnowballC)                                            # "NLP in R - pt6" ...Unnecessary?
corpus = VCorpus(VectorSource(dataset_original$Review))       # "NLP in R - pt2"
corpus = tm_map(corpus, content_transformer(tolower))         # "NLP in R - pt3"
corpus = tm_map(corpus, removeNumbers)                        # "NLP in R - pt4"
corpus = tm_map(corpus, removePunctuation)                    # "NLP in R - pt5"
corpus = tm_map(corpus, removeWords, stopwords())             # "NLP in R - pt6"
corpus = tm_map(corpus, stemDocument)                         # "NLP in R - pt7"
corpus = tm_map(corpus, stripWhitespace)                      # "NLP in R - pt8"

# Creating the Bag of Words model [§29 Lect210: "NLP in R - pt9"]
dtm = DocumentTermMatrix(corpus)
dtm = removeSparseTerms(dtm, 0.999)
dataset = as.data.frame(as.matrix(dtm))                       # "NLP in R - pt10"
dataset$Liked = dataset_original$Liked                        # "NLP in R - pt10"


# Encoding the target feature as factor                       # "NLP in R - pt10"
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set    # "NLP in R - pt10"
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set    # "NLP in R - pt10"
# install.packages('randomForest')
library(randomForest)
classifier = randomForest(x = training_set[-692],
                          y = training_set$Liked,
                          ntree = 10)

# Predicting the Test set results                             # "NLP in R - pt10"
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix                                 # "NLP in R - pt10"
cm = table(test_set[, 692], y_pred)