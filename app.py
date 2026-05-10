from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() 
                    if word not in stop_words])
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = clean_text(review)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    probability = model.predict_proba(vectorized)
    confidence = round(max(probability[0]) * 100, 2)
    sentiment = "POSITIVE 😊" if prediction[0] == 1 else "NEGATIVE 😞"
    return render_template('index.html', 
                         sentiment=sentiment,
                         confidence=confidence,
                         review=review)

if __name__ == '__main__':
    app.run(debug=True)