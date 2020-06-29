# TextPreprocessing_Spark.py
# 4/8/20
# @jessicakaye
# Used to pull subset of Amazon review data in "Clothing, Shoes & Jewelry" category and implement various text processing techniques through PySpark on GCP.



with open('contractions.py', 'w') as f:
    f.write('''CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }
    ''')

import pyspark
import sys
import os
import string
import sparknlp
import re
from contractions import CONTRACTION_MAP
import subprocess

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.ml.feature import IDF, StopWordsRemover, CountVectorizer
from pyspark.ml.clustering import LDA

# spark-nlp components. Each one is incorporated into our pipeline.
from sparknlp.annotator import *
from sparknlp.base import DocumentAssembler, Finisher

from bs4 import BeautifulSoup

# Macros
DATA_PATH = "gs://jessica-mis-586/FinalProject/Clothing_Shoes_and_Jewelry_5.json.gz"

# global variables
bucket = "jessica-mis-586"  # TODO: here, replace with your own bucket name
output_directory = 'gs://{}/hadoop/tmp/bigquery/pyspark_output/amazon_textprocessed'.format(
    bucket)

# output table and columns name
output_dataset = 'amazon'  #TODO: the name of your dataset in BigQuery
    # (Create a BigQuery dataset first using bq mk <your dataset name> in Google Cloud SDK Shell or any alternative approaches.)
output_table = 'top100products'

# Helper functions
def saveToStorage(df, output_directory, mode):
    """
    Save each df in this DStream to google storage
    Args:
        df: input df
        output_directory: output directory in google storage
        columns_name: columns name of dataframe
        mode: mode = "overwirte", overwirte the file
              mode = "append", append data to the end of file
    """
    df.write.save(output_directory, format="json", mode=mode)


def saveToBigQuery(sc, output_dataset, output_table, directory):
    """
    Put temp streaming json files in google storage to google BigQuery
    and clean the output files in google storage
    """
    files = directory + '/part-*'
    subprocess.check_call(
        'bq load --source_format NEWLINE_DELIMITED_JSON '
        '--replace '
        '--autodetect '
        '{dataset}.{table} {files}'.format(
            dataset=output_dataset, table=output_table, files=files
        ).split())
    output_path = sc._jvm.org.apache.hadoop.fs.Path(directory)
    output_path.getFileSystem(sc._jsc.hadoopConfiguration()).delete(
        output_path, True)

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

def removePunc(text):
    """
    Remove punctuation for input text.
    Args:
        text: input text.
    Returns:
        text with punctuation removed.
    """
    return text.translate(str.maketrans('', '', string.punctuation))




def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):    
    # expand contractions. I'm -> I am. Aren't -> are not
    # This is a simple version, see more elaborate version that considers word sense: https://pypi.org/project/pycontractions/
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text



def main():
    
    # Configure Spark    
    spark= SparkSession.builder\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.5")\
    .appName("amazon model")\
    .getOrCreate()
    sqlContext = SQLContext(sparkContext=spark.sparkContext, sparkSession=spark)
    
    # Need to Unzip File
    data = spark.sparkContext.textFile(DATA_PATH)
    # print data.take(5)
    
    # Read the Json as a DF
    amazon_df = sqlContext.read.json(data, multiLine=True)
    nonull_df = amazon_df.dropna(subset=('reviewText', 'overall')).select('*')
    
    review_df = (
        nonull_df.select([
            ("reviewText")
            ,("overall")
            ,("asin")
            ,("reviewerID")
            ,("reviewTime")
        ]
        )
    )
    
    # Create a column for NPS Category
    nps_category =  when(
        col("overall") == 5.0, "promoter").when(
        col("overall") == 4.0, "passive").otherwise("detractor")
        
    review_df = review_df.withColumn("nps_category", nps_category)
    
    # Create a column for Review Length
    review_df = review_df.withColumn("reviewLength", F.length("reviewText"))
    
#     # Let's see some summary stats
# #    review_df.groupBy("overall").count().orderBy("overall").show()
# #    review_df.groupBy("nps_category").count().orderBy("nps_category").show()
# #    print review_df.select("asin").distinct().count()

    # Dataset is far too large. Let's try working with something a little smaller for now.
    # We only want the top 10 products based on review count
#   sub_df = review_df.sample(False, 0.0009, seed=0)
    count_asin = review_df.groupBy("asin").count().orderBy(desc("count"))
    count_asin.show()
    top_products = count_asin.select('asin').rdd.flatMap(lambda x: x).take(100)
    
    # We want to sample the rest of the dataset for our classification training!
    train_df = review_df.where(~review_df.asin.isin(top_products))
    train_df = train_df.sample(False, 0.05, seed=0)

    # This will be all of the products we want to conduct an analysis on
#     sub_df = review_df.where(review_df.asin.isin(top_products))
        
    # Let's remove HTML tags
    htmlRemove = udf(lambda x: remove_html_tags(x))
#     sub_df = sub_df.withColumn("reviewText", htmlRemove("reviewText"))
    train_df = train_df.withColumn("reviewText", htmlRemove("reviewText"))

    # Here we want to expand contractions
    contrExpand = udf(lambda x: expand_contractions(x))
#     sub_df = sub_df.withColumn("reviewText", contrExpand("reviewText"))
    train_df = train_df.withColumn("reviewText", contrExpand("reviewText"))

    # Now we remove punctuation
    puncRemove = udf(lambda x: removePunc(x))
#     sub_df = sub_df.withColumn("reviewText", puncRemove("reviewText"))
    train_df = train_df.withColumn("reviewText", puncRemove("reviewText"))

#     sub_df.show(2)
        
    # Here we start to develop our pipeline

    # Each component here is used to some transformation to the data.
    # Up until (Finisher is part of the Spark-NLP package!)
    # The Document Assembler takes the raw text data and convert it into a format that can
    # be tokenized. It becomes one of spark-nlp native object types, the "Document".
    document_assembler = DocumentAssembler().setInputCol("reviewText").setOutputCol("document")
#     model_df = document_assembler.transform(sub_df)
    train_df = document_assembler.transform(train_df)

    
    # The Tokenizer takes data that is of the "Document" type and tokenizes it. 
    # While slightly more involved than this, this is effectively taking a string and splitting
    # it along ths spaces, so each word is its own string. The data then becomes the 
    # spark-nlp native type "Token".
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
#     model_df = tokenizer.fit(model_df).transform(model_df)
    train_df = tokenizer.fit(train_df).transform(train_df)

    # The Normalizer will group words together based on similar semantic meaning. 
    normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalizer").setLowercase(True)
#     model_df = normalizer.fit(model_df).transform(model_df)
    train_df = normalizer.fit(train_df).transform(train_df)

    # The Lemmatizer takes objects of class "Token" and converts the words into their
    # base dictionary word. For instance, the words "cars", "cars'" and "car's" would all be replaced
    # with the word "car".
    lemmatizer = LemmatizerModel.pretrained().setInputCols(["normalizer"]).setOutputCol("lemma")
#     model_df = lemmatizer.transform(model_df)
    train_df = lemmatizer.transform(train_df)

    
    # The Finisher signals to spark-nlp allows us to access the data outside of spark-nlp
    # components. For instance, we can now feed the data into components from Spark MLlib. 
    finisher = Finisher().setInputCols(["lemma"]).setOutputCols(["to_spark"]).setValueSplitSymbol(" ")
#     model_df = finisher.transform(model_df)
    train_df = finisher.transform(train_df)

    
    # Stopwords are common words that generally don't add much detail to the meaning
    # of a body of text. In English, these are mostly "articles" such as the words "the"
    # and "of".
    stopword_remover = StopWordsRemover(inputCol="to_spark", outputCol="filtered")
#     model_df = stopword_remover.transform(model_df)
    train_df = stopword_remover.transform(train_df)

    
#     # Now that our data has been pre-processed, let's move on to LDA!!
    
#     # Term frequency–inverse document frequency, is a numerical statistic that is intended 
#     # to reflect how important a word is to a document in a collection or corpus
    
#     # First we'll do the term frequency using CountVectorizer
#     # TF (term frequency) creates a matrix that counts how many times each word in the
  #     # vocabulary appears in each body of text. This then gives each word a weight based
  #     # on its frequency.
  #     tf = CountVectorizer(inputCol="filtered", outputCol="raw_features")
  #     tf_model = tf.fit(model_df)
  #     model_df = tf_model.transform(model_df)
  #     vocab = tf_model.vocabulary

  #     # Here we implement the IDF portion. IDF (Inverse document frequency) reduces 
  #     # the weights of commonly-appearing words. 
  #     idf = IDF(inputCol="raw_features", outputCol="features")
  #     model_df = idf.fit(model_df).transform(model_df)
      
  #     model_df.printSchema()
          
  #     saveToStorage(model_df, output_directory, mode="overwrite")
  #     saveToBigQuery(spark.sparkContext, output_dataset, output_table, output_directory)
  #     saveToStorage(model_df, 'gs://jessica-mis-586/AmazonData/text_processed.json', mode="overwrite")
      saveToStorage(train_df, 'gs://jessica-mis-586/AmazonData/train_processed', mode="overwrite")


if __name__ == "__main__":
    main()
