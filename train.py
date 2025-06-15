import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# --- Data Loading ---
print("🔍 Loading data...")
try:
    df = pd.read_csv("job_postings.csv")  # Replace with your file path
    print("Data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# --- Text Cleaning ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove special chars/numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

print("🧹 Cleaning text...")
df['cleaned_text'] = df['text_column'].apply(clean_text)  # Replace 'text_column' with your column name

# --- EDA ---
print("\n📊 Class Distribution:")
print(df['is_scam'].value_counts(normalize=True))  # Replace 'is_scam' with your target column

# --- Vectorization ---
print("\n⚙️ Vectorizing text...")
tfidf = TfidfVectorizer(
    min_df=1,      # Minimum 1 document
    max_df=0.9,     # Ignore terms in >90% docs
    ngram_range=(1, 2)  # Use unigrams and bigrams
)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['is_scam']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---
print("\n⚙️ Training model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# --- Evaluation ---
print("\n📊 Model Performance:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# --- Feature Importance ---
plt.figure(figsize=(10, 6))
sns.barplot(
    x=model.feature_importances_[:20],
    y=tfidf.get_feature_names_out()[:20]
)
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\n✅ Training completed! Feature importance plot saved.")