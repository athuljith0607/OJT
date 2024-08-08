import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('IMDB Dataset.csv')

# Function to clean text data
def clean_text(text):
    text = re.sub(r'<br />', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation and numbers
    text = text.lower()  # Convert to lowercase
    return text

# Clean the reviews
data['review'] = data['review'].apply(clean_text)

# Encode the sentiment labels
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Function to make predictions on new reviews
def predict_sentiment(reviews):
    reviews_clean = [clean_text(review) for review in reviews]
    reviews_tfidf = tfidf_vectorizer.transform(reviews_clean)
    predictions = classifier.predict(reviews_tfidf)
    sentiments = label_encoder.inverse_transform(predictions)
    return sentiments

# Example usage: Predict sentiment of new reviews
new_reviews = [
    "I absolutely loved this movie! The acting was great and the plot was so engaging.",
    "This was a terrible film. I wasted two hours of my life watching this."
]
predictions = predict_sentiment(new_reviews)
print(f'New Reviews: {new_reviews}')
print(f'Predictions: {predictions}')