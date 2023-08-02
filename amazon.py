from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#Create spark session
spark = SparkSession.builder.appName("AmazonNormalize").getOrCreate()

# download nltk
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

# clean the text data
def clean_text (text):
    if text is None:
        return None
    # lowercase text
    text = text.lower()
  
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
  
    # remove special characters
    text = re.sub(r'[^A-Za-zÀ-ú ]+', '', text)
    text = re.sub('book|one', '', text)
  
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
  
    # tokenized words and remove stop words 
    words = [word for word in word_tokenize(text) if word not in stop_words]
  
    #stemming
    porter = PorterStemmer()
    text = " ".join([porter.stem(word) for word in words])
    return text

clean_text_udf = udf(clean_text, StringType())

# load data
df = spark.read.csv("hdfs://master:9000/user/hadoop/amazon_review_polarity_csv/train.csv")

# renaming columns
df = df.withColumnRenamed("_c0", "sentiment").withColumnRenamed("_c1", "title").withColumnRenamed("_c2", "reviews")

# filter only negative sentiment review
df = df.filter(df.sentiment == 1)

# drop columns 
df = df.drop("sentiment", "title")

# Drop nul values
df = df.dropna(subset = 'reviews')

# clean text
df = df.withColumn("clean_reviews", clean_text_udf(col("reviews")))

# save normalized data
df.write.csv("hdfs://master:9000/user/hadoop/clean_reviews.csv", header = True)

spark.stop()
