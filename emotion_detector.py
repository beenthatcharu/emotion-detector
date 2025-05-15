import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# === STEP 1: Load Data ===
def load_data(file_path):
    return pd.read_csv(file_path, sep=';', names=['text', 'label'])

train_df = load_data("train.txt")
test_df = load_data("test.txt")

# === STEP 2: Clean Text ===
train_df['clean_text'] = train_df['text'].apply(nfx.remove_punctuations)
test_df['clean_text'] = test_df['text'].apply(nfx.remove_punctuations)

# === STEP 3: Prepare Features ===
X_train = train_df['clean_text']
y_train = train_df['label']
X_test = test_df['clean_text']
y_test = test_df['label']

# === STEP 4: Vectorize Text ===
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === STEP 5: Train Model ===
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# === STEP 6: Evaluate ===
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# === STEP 7: Save Model and Vectorizer ===
joblib.dump(model, "emotion_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# === STEP 8: Define Predict Function ===
def predict_emotion(text):
    text_clean = nfx.remove_punctuations(text)
    vect = vectorizer.transform([text_clean])
    return model.predict(vect)[0]

# === STEP 9: Try a Prediction ===
test_input = "I can't believe how amazing this day is!"
print("Predicted Emotion:", predict_emotion(test_input))
while True:
    user_input = input("\nEnter a sentence (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("Predicted Emotion:", predict_emotion(user_input))


