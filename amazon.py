from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lower, regexp_replace, concat_ws
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.feature import Tokenizer
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.corpus import stopwords

# Create Spark session
spark = SparkSession.builder.appName("AmazonNormalize").getOrCreate()

# Download nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean the text data
def clean_text(words):
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    # Stemming
    porter = PorterStemmer()
    words = [porter.stem(word) for word in words]
    return words

clean_text_udf = udf(clean_text, ArrayType(StringType()))

# Load data
df = spark.read.csv("hdfs://master:9000/user/hadoop/amazon_review_polarity_csv/train.csv")

# Renaming columns
df = df.withColumnRenamed("_c0", "sentiment").withColumnRenamed("_c1", "title").withColumnRenamed("_c2", "reviews")

# Filter only negative sentiment review
df = df.filter(df.sentiment == 1)

# Drop columns 
df = df.drop("sentiment", "title")

# Lowercase text, remove punctuation and special characters
df = df.withColumn("reviews", lower(col("reviews")))
df = df.withColumn("reviews", regexp_replace(col("reviews"), r'[^\w\s]', ''))
df = df.withColumn("reviews", regexp_replace(col("reviews"), r'[^A-Za-zÀ-ú ]+', ''))
df = df.withColumn("reviews", regexp_replace(col("reviews"), 'book|one', ''))
df = df.withColumn("reviews", regexp_replace(col("reviews"), r'\s+', ' '))

#filter out null values in reviews
df = df.filter(df.reviews.isNotNull())

# Tokenize words
tokenizer = Tokenizer(inputCol="reviews", outputCol="words")
df = tokenizer.transform(df)

# Clean text
df = df.withColumn("clean_reviews", clean_text_udf(col("words")))

#drop column word
df = df.drop("words")

# join word into a string
df = df.withColumn("clean_reviews", concat_ws(" ", col("clean_reviews")))

# remove reviews
df = df.drop("reviews")

# Save normalized data
df.write.mode('overwrite').csv("hdfs://master:9000/user/hadoop/clean_data.csv", header = True)

#show few outputs
df.show(20, truncate=False)

spark.stop()
