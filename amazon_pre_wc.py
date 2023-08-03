import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import download
from collections import Counter

# Download NLTK 
download('punkt')
download('stopwords')

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Get stopwords
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    # Remove punctuations and numbers
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    word_tokens = word_tokenize(text)

    # Stopword removal and stemming
    text = [stemmer.stem(word) for word in word_tokens if word not in stop_words]

    text = ' '.join(text)
    
    return text

# Load the data
data = pd.read_csv('~/newworkspace/amazon_review_polarity_csv/train.csv', header=None, names=['sentiment', 'title', 'review'])

# Fill NaNs with an empty string
data = data.fillna("")

# Apply the function to the 'title' and 'review' columns
data['title'] = data['title'].apply(clean_text)
data['review'] = data['review'].apply(clean_text)

# Select only negative reviews
data = data[data['sentiment'] == 1]

# Remove null values
data = data.dropna()

# Keep only the cleaned 'review' column
data = data['review']

# Reset the index of the DataFrame
data = data.reset_index(drop=True)

# Split the reviews into words and count the occurrences of each word
word_counts = Counter(' '.join(data).split())

# Convert the word counts to a DataFrame and reset the index
word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index').reset_index()

# Rename the columns
word_counts_df.columns = ['Word', 'Count']

# Sort the DataFrame by the counts in descending order
word_counts_df = word_counts_df.sort_values(by='Count', ascending=False)

# Print the top 10 words
print(word_counts_df.head(10))

# Save output to a CSV file
word_counts_df.to_csv('~/newworkspace/word_counts.csv', index=False)
