# Spotify_popularity_classification.py
# 4/24/20
# @jessicakaye
# Used to conduct classification for popularity label based on song features and other decided numeric features
# Algorithms include KNN, Decision Tree (CART), Random Forest, and Naive Bayes*** with option to include Logistic Regression and SVM
# ***PCA conducted prior to Naive Bayes

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import seaborn as sns

pd.set_option('display.max_columns', None)


spotify_df = pd.read_csv('processed_spotify_data_3-22.csv')

print(spotify_df.describe())

ax = sns.countplot(x='popularity_class_label', data = spotify_df, palette = 'Set1')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Popularity', fontsize=12)
total = float(len(spotify_df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            '{0:.0%}'.format(height/total),
            ha="center")

variables= ['danceability','energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
           'instrumentalness','liveness', 'valence', 'tempo', 'refined_artist_popularity',
            'duration_ms', 'time_signature', 'days_since_release', 'contains_features?',
           'is_single', 'is_album', 'is_compilation']

# Here we determine that there is a correlation between some variables... Not good. We need to fix this later!
plt.figure()
sns.heatmap(spotify_df[variables].corr(), annot=True, fmt='.1g', cmap='coolwarm')
plt.tight_layout()

#Is our data linearly separable?
spotify_df['Popular'] = pd.get_dummies(spotify_df['popularity_class_label'], drop_first=True)

x = spotify_df[variables]
y = spotify_df['Popular']
sc = StandardScaler()
x = sc.fit_transform(x)

svm = svm.SVC(C=1.0, kernel='linear', random_state=0)
svm.fit(x, y)

predicted = svm.predict(x)

cm = confusion_matrix(y, predicted)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative', 'Positive']
plt.title('SVM Linear Kernel Confusion Matrix - Popular')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN', 'FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][1]+cm[1][0]))

Let's try using t-Distributed Stochastic Neighbouring Entities (t-SNE)
t-SNE looks at how to best represent data by looking at:
a distribution that measures pairwise similarities of the input objects and
a distribution that measures pairwise similarities of the corresponding low-dimensional points in the embedding
This technique is best for VISUALIZING YOUR DATA ONLY. Does not work with any density or distance-based algorithm

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
tsne_result = tsne.fit_transform(spotify_df[variables])
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

tsne_result_df = pd.DataFrame(data=tsne_result)
tsne_df = pd.concat([tsne_result_df, spotify_df['popularity_class_label']], axis = 1)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x=0, y=1,
    hue="popularity_class_label",
    palette=sns.color_palette("Set1", 2),
    data=tsne_df,
    legend="full",
    alpha=1
)
plt.title('t-SNE graph for dimensionality reduction')


# create a sample set based on 50% of the dataset.
test_df = spotify_df.sample(frac = 0.5, replace = False, random_state = 1)
test_df.reset_index(inplace=True)

# Now we should separate out the features
X = test_df[variables]
y = test_df['popularity_class_label']

# we want to get binary labels for the popularity
y = pd.get_dummies(y, drop_first=True)

# We now want to split our data into training and testing.
# By default, train_test_split automatically splits to 70-30 training to testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

# Let's STANDARDIZE our data!
std = StandardScaler()
std_X_train = pd.DataFrame(std.fit_transform(X_train), columns = variables)
std_X_test = pd.DataFrame(std.transform(X_test), columns = variables)

# Let's normalize our data!
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = variables)
X_test = pd.DataFrame(scaler.transform(X_test), columns = variables)

# We need to use PCA to reduce multicollinearity!
# Using n_component = 2 seems to give us the highest accuracy and the best recall, but we want to
# preserve 99% of the variance!
# Keep in mind: PCA looks for attributes with the most variance
pca = PCA(n_components=0.99)
pca_result = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)
principal_df = pd.DataFrame(data=pca_result)
pca_df = pd.concat([principal_df, y_train], axis = 1)

# The explained_variance ratio provides us the amount of information that each principal component holds!
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# We want to utilize dummy classifiers to determine a proper baseline accuracy.
strategies = ['most_frequent', 'stratified', 'uniform', 'constant']

test_scores = []
for s in strategies:
    if s == 'constant':
        dclf = DummyClassifier(strategy=s, random_state=0, constant=1)
    else:
        dclf = DummyClassifier(strategy=s, random_state=0)
    dclf.fit(X_train, y_train)
    score = dclf.score(X_test, y_test)
    test_scores.append(score)

print("Baseline Accuracy values based on Most Frequent, Stratified, Uniform, & Constant "
      "Dummy Classifiers: ")
print(test_scores)

# Here we want to show the baseline accuracy for the different strategies used
plt.figure(figsize=(16,10))
ax = sns.stripplot(strategies, test_scores);
ax.set(xlabel='Strategy', ylabel='Test Accuracy')

# We need to create lists to compare the model evaluation metrics
accuracy = []
sensitivity = []
specificity = []
precision = []
f1 = []
# We now create an instance of the classifier. The default value of k is 5.
# In general practice, the value of k = sqrt(N) where N is # samples in training set and is odd for binary classes.
# Here it will be 81 according to the statement below.
# print(round(math.sqrt(X_train.shape[0])))
knn = KNeighborsClassifier(n_neighbors=81, metric='euclidean')
knn.fit(X_train, y_train.values.ravel())

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train.values.ravel())

nb = GaussianNB()
nb.fit(principal_df, y_train.values.ravel())


# logreg = LogisticRegression(max_iter = 1000)
# logreg.fit(principal_df, y_train.values.ravel())
#
# svm = svm.SVC(C=1.0, kernel='linear', random_state=0, probability=True)
# svm.fit(X_train, y_train.values.ravel())


classifiers = [knn, dt, rf, nb]#logreg, svm]
classifiers_labels = ['k-Nearest Neighbors', 'Decision Tree CART', 'Random Forest', 'Naive Bayes']
                      # 'Logistic Regression','Linear Support Vector Machines']
cmatrix_list = []

for c in classifiers:
    if c == nb:
        y_pred = c.predict(pca_X_test)
    else:
        y_pred = c.predict(X_test)

    #let's create a confusion matrix that we can then display as a heatmap
    cfm = confusion_matrix(y_test, y_pred)
    cfmatrix = pd.DataFrame(cfm, range(2),range(2))
    cfmatrix.columns =['predicted positive', 'predicted negative']
    cfmatrix.index = ['actual positive', 'actual negative']
    print(cfmatrix)

    cmatrix_list.append(cfmatrix)

    # Here we evaluate our model
    accuracyc = (cfm[0][0] + cfm[1][1])/(y_test.shape[0])
    # what is the completeness or performance of the algorithm to predict positives?
    sensitivityc = (cfm[0][0])/(cfm[0][0]+cfm[0][1]) #TP/TP+FN
    # what is the performance of the algorithm to predict negatives?
    specificityc = (cfm[1][1])/(cfm[1][0]+cfm[1][1]) # TN/FP+TN
    # what is the exactness of the results?
    precisionc = cfm[0][0]/(cfm[0][0]+cfm[1][0]) # TP/TP+FP
    # RECALL AND SENSITIVITY ARE THE SAME THING **
    f1c = 2 * precisionc * sensitivityc / (precisionc + sensitivityc)

    accuracy.append(accuracyc)
    sensitivity.append(sensitivityc)
    specificity.append(specificityc)
    precision.append(precisionc)
    f1.append(f1c)

# This is the dataframe that compares all of the classifiers together!!
classifier_df = pd.DataFrame(zip(accuracy, sensitivity, specificity, precision, f1),
                             columns = ['Accuracy', 'Sensitivity/Recall', 'Specificity', 'Precision', 'F1 Score'],
                             index=classifiers_labels)
classifier_df['Support'] = sum(y_test['Popular'] == 1)

print(classifier_df)

# Here we want to show all of the different correlation matrices
for idx, val in enumerate(cmatrix_list):
    plt.figure()
    sns.heatmap(val, annot=True, fmt='d', cmap = 'YlGn')
    plt.title(classifiers_labels[idx] + ' Classifier Confusion Matrix')


# Here is another way to view the accuracy of the classifiers: ROC Curve
# The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible
plt.figure()
for idx, cls in enumerate(classifiers):
    if classifiers[idx] == nb:
        classifier_roc_auc = roc_auc_score(y_test, cls.predict(pca_X_test))
        fpr, tpr, thresholds = roc_curve(y_test, cls.predict_proba(pca_X_test)[:, 1])
    else:
        classifier_roc_auc = roc_auc_score(y_test, cls.predict(X_test))
        fpr, tpr, thresholds = roc_curve(y_test, cls.predict_proba(X_test)[:, 1])

    plt.plot(fpr, tpr, label=(classifiers_labels[idx] + ' (area = %0.2f)') % classifier_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curves')
    plt.legend(loc="lower right")

# Our best performing classifier was Random Forest. Let's see the most important features!
importances = rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

feature_names = [variables[i] for i in indices]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names)
plt.xlim([-1, X.shape[1]])
plt.ylabel("importance", fontsize = 12)
plt.xlabel("feature", fontsize = 12)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
