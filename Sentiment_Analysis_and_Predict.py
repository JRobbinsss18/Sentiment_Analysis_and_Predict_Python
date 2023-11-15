import re
import pandas as pd 
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize
#import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import seaborn as sns
from sklearn.metrics import confusion_matrix

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
custom_stopwords = ['the', 'and', 'is', 'in', 'at', 'which', 'on', 'for', 'with', 'it', 'an', 'this', 'that', 'by', 'from', 'as', 'are', 'was', 'be', 'to', 'of', 'a', 'or', 'but', 'had', 'not', 'we', 'they', 'you', 'he', 'she', 'i', 'my', 'your', 'our', 'their', 'his', 'her', 'its', 'me', 'us', 'them', 'him', 'hers']
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Tokenize the text into words
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in custom_stopwords and len(word) > 2]
    text = ' '.join(words)
    return text

def generate_wordcloud(text, filename):
    wordcloud = WordCloud(width = 800, height = 800,
                          background_color ='white',
                          min_font_size = 10).generate(text)

    # Save wordcloud to file
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig(filename, format='png')
    plt.close()

def save_confusion_matrix(y_true, y_pred, filename, labels=['Negative', 'Positive'], title='Confusion Matrix'):
    # Generating the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plotting the confusion matrix
    plt.figure(figsize=(10,7))
    sns_heatmap = sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(title)
    plt.savefig(filename, format='png')
    plt.close()


# read in csv file
df = pd.read_csv('Training_IMDB_Dataset.csv')
# print first few rows of data
print(df.head())
df['review'] = df['review'].apply(preprocess_text)
#for i, review in enumerate(df['review']):
#    df.at[i, 'review'] = preprocess_text(review, i)
print(df.head())

# Concatenate all reviews with positive sentiment
positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['review'])
# Concatenate all reviews with negative sentiment
negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['review'])
generate_wordcloud(positive_reviews, 'positive_reviews_wordcloud.png')
generate_wordcloud(negative_reviews, 'negative_reviews_wordcloud.png')


#GETTING INTO PREDICTION/MACHINE LEARNING HERE!
# Vectorizing the text data
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['review'])
y = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Reading the new dataset
test_df = pd.read_csv('test.csv')

# Preprocessing the text in the new dataset
test_df['text'] = test_df['text'].apply(preprocess_text)

# Vectorizing the text data using the same TF-IDF vectorizer
X_new = tfidf.transform(test_df['text'])
y_new_true = test_df['sentiment'].apply(lambda x: 1 if x == 'pos' else 0)

# Predicting the sentiment
y_new_pred = model.predict(X_new)

# Evaluating the model on the new dataset
print("Accuracy on new dataset:", accuracy_score(y_new_true, y_new_pred))
print(classification_report(y_new_true, y_new_pred))

save_confusion_matrix(y_new_true, y_new_pred, 'confusion_matrix.png')
