import nltk

def setup():
    nltk.download('punkt')
    nltk.download('stopwords')

if __name__ == "__main__":
    setup()
    print("NLTK data installed successfully!")