import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from time import time
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud

pd.set_option('display.max_columns', None)

# Load the dataset!
df = pd.read_json('AmazonData_text_processed_full.json', lines = True)
print(df)
print(df.describe())

# Let's drop those duplicates
df.drop_duplicates(['overall', 'reviewText', 'reviewTime', 'asin', 'reviewerID'], inplace=True)

#plot for all of the products
plt.figure(figsize=(16,10))
ax = sns.countplot(x='asin', data = df, palette = 'Set1', order=df['asin'].value_counts().index)
plt.xlabel('ASIN', fontsize=12)
plt.ylabel('Count', fontsize=12)
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 10,
            '{}'.format(height),
            ha="center")

plt.title("Count of Reviews Per ASIN")
plt.savefig("Count of Reviews Per ASIN.png")

#Distribution of Ratings!
plt.figure()
ax = sns.countplot(x='overall', data=df, palette='Set1', order=df['overall'].value_counts().index)
plt.xlabel('overall', fontsize=12)
plt.ylabel('Count', fontsize=12)
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 10,
            '{0:.0%}'.format(height / total),
            ha="center")
plt.title("Count of Reviews Per Rating")
plt.savefig("Count of Reviews Per Rating.png")

# Distribution of NPS Categories!
plt.figure()
ax = sns.countplot(x='nps_category', data=df, palette='Set1', order=df['nps_category'].value_counts().index)
plt.xlabel('nps_category', fontsize=12)
plt.ylabel('Count', fontsize=12)
total = float(len(df))
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2.,
            height + 10,
            '{0:.0%}'.format(height / total),
            ha="center")
plt.title("Count of Reviews Per NPS Category")
plt.savefig("Count of Reviews Per NPS Category.png")

# Let's create a wordcloud!
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(df['filtered'].to_string())

# plot the wordcloud!
plt.figure(figsize=(16,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.savefig('wordcloudoftop10products')


# Let's optimize our df and try using CountVectorizer
# I already have these columns from text processing in Spark, but I want to try the following in sklearn
amazon_df = df.drop(labels=['raw_features', 'features'], axis=1)


# Let's create a list of all of the different ASINs
list_asins =  amazon_df.asin.unique()


sns.set_style('whitegrid')

# Helper function
def plot_10_most_common_words(asin, count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title=f'10 most common words for {asin}')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.tight_layout()

    plt.savefig(f'{asin}_topwords.png')

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def topics_words(model, feature_names, n_top_words):
    topics = []
    words =[]
    for topic_idx, topic in enumerate(model.components_):
        topics.append(topic_idx)
        words.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
    new_df = pd.DataFrame(list(zip(topics, words)), columns=['topicID', 'words'])
    return new_df



n_top_words = 6
n_components = 7

all_words_and_topics = pd.DataFrame(columns=['topicID', 'words', 'asin', 'num documents'])
all_asins_df = pd.DataFrame(columns=list(amazon_df.columns.values))

# We want to find the top words per product. Let's create a loop.
for asin in list_asins[0:5]:
    asin_df = amazon_df.loc[amazon_df['asin'] == str(asin)]
    asin_df.reset_index(inplace=True)
    # Initialise the count vectorizer with the English stop words
    # We are going to use the raw term count for LDA
    print("Extracting tf features for LDA...")
    stop_words = ENGLISH_STOP_WORDS
    cv = CountVectorizer(stop_words='english', analyzer=lambda x:[w for w in x if w not in stop_words])
    # Fit and transform the processed titles
    t0 = time()
    count_vector = cv.fit_transform(asin_df['filtered'])
    print("done in %0.3fs." % (time() - t0))
    print()

    # Materialize the sparse data
    data_dense = count_vector.todense()

    # Compute Sparsicity = Percentage of Non-Zero cells
    print("Sparsicity: ", ((data_dense > 0).sum() / data_dense.size) * 100, "%")

    # Visualise the 10 most common words
    plot_10_most_common_words(asin, count_vector, cv)

    print("Fitting LDA models with tf features...")
    lda = LatentDirichletAllocation(n_components=n_components, learning_method='online')
    t0 = time()
    # This is the Document - Topic Matrix
    lda_output = lda.fit_transform(count_vector)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in LDA model:")
    tf_feature_names = cv.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

    # Log Likelihood: Higher the better
    print("Log Likelihood: ", lda.score(count_vector))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda.perplexity(count_vector))

    # See model parameters
    # print(lda.get_params())

    # column names
    topicnames = ["Topic" + str(i) for i in range(lda.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(asin_df.shape[0])]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames)#, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic_weight'] = np.amax(df_document_topic, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    print(df_document_topic)

    asin_df = asin_df.join(df_document_topic['dominant_topic'].astype('int'), how = 'inner')
    asin_df = asin_df.join(df_document_topic['dominant_topic_weight'], how='inner')
    all_asins_df = pd.concat([all_asins_df, asin_df])

    #What is the topic distribution across documents?
    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="num documents")
    df_topic_distribution.columns = ['topicID', 'num documents']
    print(df_topic_distribution)

    asintw = topics_words(lda, tf_feature_names, n_top_words)
    asintw['asin'] = asin
    asintw = asintw.merge(df_topic_distribution, on = "topicID", how = "inner")
    all_words_and_topics = pd.concat([all_words_and_topics, asintw])

print(all_words_and_topics)
print(all_asins_df)


all_asins_df.to_csv('all_asins_and_indices.csv')
all_words_and_topics.to_csv('all_words_and_topics.csv')
#
#
# # plt.show()
