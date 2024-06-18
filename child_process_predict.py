from unidecode import unidecode
import joblib
import sys
import numpy as np
import json


model = joblib.load('self_training_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
category_le = joblib.load('label_encoder.joblib')

posts_to_categorize = json.loads(sys.stdin.read())

textsList = []
for idx, post in enumerate(posts_to_categorize):
    post_text = post.get('text', '')
    newline = '\n' if post_text else ''
    attached_post_text = post['attachedPost'].get('text', '') if post.get('attachedPost') else ''
    text = f"{post_text}{newline}{attached_post_text}"
    textsList.append(unidecode(text))


X = vectorizer.transform(textsList)

y_pred_proba = model.predict_proba(X)

for idx in range((len(posts_to_categorize))):
    prediction_dict = {category: prob for category, prob in zip(category_le.classes_, y_pred_proba[idx])}
    category = category_le.inverse_transform([np.argmax(y_pred_proba[idx])])[0]

    posts_to_categorize[idx]['predict'] = {
        'proba': prediction_dict,
        'category': category
    }

print(json.dumps(posts_to_categorize))