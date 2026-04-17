# 32_saad_assignment-2
Sentiment Analysis of Tweets Using Machine Learning
1. Introduction (

This project focuses on performing sentiment analysis on tweets related to a specific topic. The chosen topic for this study is public sentiment expressed on Twitter (X) regarding trending discussions.

The topic was selected because social media platforms like Twitter are rich sources of real-time opinions, making them highly relevant for sentiment analysis tasks. Understanding these sentiments can help in areas such as brand monitoring, public opinion tracking, and decision-making.
Data Collection

A dataset of 100 tweets was collected and stored in a CSV file named shuffled_tweets_dataset.csv. The tweets were gathered using

Manual collection from Twitter

Each tweet was labeled manually into one of three sentiment categories:

Positive
Neutral
Negative

Manual tagging was done by reading each tweet carefully and assigning sentiment based on the expressed opinion.

Data Preprocessing

Before training the models, the text data was cleaned and preprocessed using the following steps:

Conversion of all text to lowercase
Removal of URLs (links)
Removal of mentions (@username) and hashtags (#)
Removal of punctuation and special characters
Tokenization (splitting text into words)
Removal of stopwords using NLTK
Preprocessing Function (Code Snippet):
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return " ".join(filtered_tokens)

After preprocessing, a new column cleaned_text was created in the dataset.

3. Data Splitting

The dataset was divided into:

80% Training Data (80 tweets)
20% Testing Data (20 tweets)

This was implemented using the train_test_split function:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
Justification:

The 80–20 split is a standard practice in machine learning as it provides:

Sufficient data for training the model
Enough unseen data to fairly evaluate model performance
4. Model Training & Classification 
Classifiers Used

The following machine learning models were used:

Naïve Bayes (MultinomialNB)
Support Vector Machine (SVM)
Logistic Regression
Feature Extraction

Text data was converted into numerical form using TF-IDF Vectorization:

tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
Training Process

Each model was trained using the TF-IDF features of the training dataset:

models = {
    "Naïve Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear'),
    "Logistic Regression": LogisticRegression()
}

results = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    print(f"\n--- {name} Performance ---")
    print(classification_report(y_test, predictions))
    results[name] = accuracy_score(y_test, predictions)
Hyperparameters Used
Naïve Bayes: Default parameters
SVM: Linear kernel (kernel='linear')
Logistic Regression: Default settings
Model Evaluation
Accuracy score was used to compare models
Classification report provided precision, recall, and F1-score
Confusion matrix visualized prediction performance

The best-performing model was selected using:

best_model = max(results, key=results.get)
print(f"Best Performing Model: {best_model}")
