# Amazon_Classification.py
# 4/28/20
# @jessicakaye
# Used to conduct sentiment analysis on Amazon reviews for the top 5 most reviewed products
# using Linear Support Vector Machines



import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from time import time

from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from time import time

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)

# Load the dataset!
df = pd.read_json('AmazonData_text_processed_full.json', lines = True)

# Let's drop those duplicates
df.drop_duplicates(['overall', 'reviewText', 'reviewTime', 'asin', 'reviewerID'], inplace=True)
# Let's optimize our df
# I already have these columns from text processing in Spark, but I want to try the following in sklearn
df = df.drop(labels=['raw_features', 'features'], axis=1)
df.loc[df['nps_category'] == 'promoter', 'nps_dummy'] = 1
df.loc[df['nps_category'] == 'detractor', 'nps_dummy'] = -1
df.loc[df['nps_category'] == 'passive', 'nps_dummy'] = 0
print(df.describe())


# This is the df with the top 5 ASINs
top_5_df = pd.read_csv('all_asins_and_indices.csv')
top_5_df = top_5_df.rename(columns={'Unnamed: 0': 'index', 'index': 'og_index'})
top_5_df.astype({'index': 'int64'})
print(top_5_df.describe())
print(top_5_df.info())

# Create a list of the top 5 ASINS for later reference
top_asins_list = top_5_df.asin.unique()

# Create a density plot comparing the distribution of dominant topic weights for each ASIN
plt.figure()
for asin in top_asins_list:
    sns.distplot(top_5_df['dominant_topic_weight'].loc[top_5_df['asin'] == asin], hist = False, kde = True,
                     kde_kws = {'linewidth': 2}, label = asin)
plt.title('Distribution of Dominant Topic Weights per ASIN')
plt.savefig('Distribution of Dominant Topic Weights per ASIN')

# Clean up the df with the words and topics per ASIN
words_topics_df = pd.read_csv('all_words_and_topics.csv')
words_topics_df = words_topics_df.rename(columns={'Unnamed: 0': 'index'})
words_topics_df.astype({'index': 'int64'})
print(words_topics_df.describe())
print(words_topics_df.info())

# NOW, let's do the actual sentiment analysis!!
# Define our parameters
others_df = df.loc[(~df['asin'].isin(top_asins_list))]
X = others_df['filtered']
y = others_df['nps_category']
print(X)
print(y)

# Let's use TF-IDF Vectorization for input to SVM
def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None)

# Fit and transform the processed titles
tfidf_vector = tfidf.fit_transform(X)

# summarize
print('vocabulary: ', tfidf.vocabulary_)

# Let's split our data into training and testing.
# We want to TRAIN and TEST on all the other reviews for the ASINs that are not in the top 5
X_train, X_test, y_train, y_test = train_test_split(tfidf_vector, y, test_size = 0.2, random_state = 1)

# FINALLY, time to create the classification model!!
# Perform classification with SVM, kernel=linear
svm_cls = svm.SVC(kernel='linear')
t0 = time()
svm_cls.fit(X_train, y_train)
t1 = time()
y_pred = svm_cls.predict(X_test)
t2 =time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(y_test, y_pred, target_names=others_df.nps_category.unique())
print(report)

# let's create a confusion matrix that we can then display as a heatmap
cfm = confusion_matrix(y_test, y_pred)
cfmatrix = pd.DataFrame(cfm, range(3), range(3))
cfmatrix.columns = ['predicted positive', 'predicted neutral','predicted negative']
cfmatrix.index = ['actual positive', 'actual neutral', 'actual negative']
print(cfmatrix)

plt.figure()
sns.heatmap(cfmatrix, annot=True, fmt='d', cmap='YlGn')
plt.title('Linear SVM Classifier Confusion Matrix')
plt.savefig('Linear SVM Confusion Matrix')

# Here we evaluate our model
accuracyc = (cfm[0][0] + cfm[1][1] + cfm[2][2]) / (y_test.shape[0])

print(f"""
accuracy: {accuracyc}
""")

# # pickling the vectorizer
# pickle.dump(tfidf, open('tfidf.sav', 'wb'))
# # pickling the model
# pickle.dump(svm_cls, open('classifier.sav', 'wb'))
#
# columns = list(top_5_df.columns.values) + ['predicted', 'LDA_filter?']
#
# # Here we create a loop where we gauge the overall sentiment of the product based on the sentiment from the reviews
# sentiment_analysis_asins_df = pd.DataFrame(columns=columns)
#
# for asin in top_asins_list:
#     asin_df = top_5_df.loc[top_5_df['asin'] == str(asin)]
#     asin_df.reset_index(inplace=True)
#     # define our variables
#     asin_X = asin_df['filtered']
#     asin_y = asin_df['nps_category']
#     # tf-idf for the features
#     asin_X_vector = tfidf.transform(asin_X)
#     # run the classifier
#     asin_y_pred = svm_cls.predict(asin_X_vector)
#
#     # now we create the output!
#     # add the predicted values of the label based on the classifier
#     asin_y_pred = pd.DataFrame(asin_y_pred, columns={'predicted'})
#
#     asin_df = pd.concat([asin_df, asin_y_pred], axis=1)
#     print(asin_df)
#     # We conducted LDA on the reviews for this product. We need to create a label that dictates whether or not
#     # a review contains a high enough 'dominant_topic_weight' to be considered as a review with a specialty topic
#     # For these reviews that do have a high enough dominant topic weight, we will filter based on it to see how
#     # sentiment changes!
#     asin_df.loc[asin_df['dominant_topic_weight'] < asin_df['dominant_topic_weight'].median(), 'LDA_filter?'] = False
#     asin_df.loc[asin_df['dominant_topic_weight'] >= asin_df['dominant_topic_weight'].median(), 'LDA_filter?'] = True
#
#     sentiment_analysis_asins_df = pd.concat([sentiment_analysis_asins_df, asin_df])
#
# print(sentiment_analysis_asins_df)
# sentiment_analysis_asins_df.to_csv('top5asins_sentimentanalysis.csv')
#
