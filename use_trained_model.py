from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import json
import joblib

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
posts_collection = client['JV-FINDER']['posts']


model = joblib.load('random_forest_model-94.61.joblib')
vectorizer = joblib.load('tfidf_vectorizer-94.61.joblib')
category_le = joblib.load('label_encoder.joblib')

post_to_categorize = list(posts_collection.aggregate([
        { '$match': { 'manualDefinition': { "$exists": False } } },
        { '$sample': { 'size': 1 } }
    ]))[0]

post_text = post_to_categorize.get('text', '')
newline = '\n' if post_text else ''
attached_post_text = post_to_categorize['attachedPost'].get('text', '') if post_to_categorize.get('attachedPost') else ''
text = f"{post_text}{newline}{attached_post_text}"

print(text)

X = vectorizer.transform([text])

y_pred = model.predict_proba(X)[0]

labels = category_le.inverse_transform(np.arange(len(y_pred)))

label_probabilities = dict(zip(labels, y_pred * 100))

print(label_probabilities)

# print(category_le.inverse_transform(y_pred))