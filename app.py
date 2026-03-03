import sys
import pickle

with open("models/classifier.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

text = sys.argv[1]
vec = vectorizer.transform([text])
prediction = model.predict(vec)[0]

print(prediction)