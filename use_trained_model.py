from pymongo import MongoClient
from dotenv import load_dotenv
import numpy as np
import os
import json
import joblib
from openai import OpenAI
from unidecode import unidecode

from pprint import pprint

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
posts_collection = client['JV-FINDER']['posts']

openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


model = joblib.load('self_training_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
category_le = joblib.load('label_encoder.joblib')

# feature_importances = model.base_estimator_.feature_importances_
# feature_names = vectorizer.get_feature_names_out()

# important_features = [(feature_names[i], feature_importances[i]) for i in range(len(feature_importances))]

# important_features.sort(key=lambda x: x[1], reverse=False)

# for feature, importance in important_features:
#     if (importance > 0):
#         print(f"Feature: {feature}, Importance: {importance}")

# print(sum(1 for _, importance in important_features if importance > 0))

# posts_to_categorize = list(posts_collection.aggregate([
#         { '$match': { 'manualDefinition': { "$exists": False } } },
#         { '$sample': { 'size': 200 } }
#     ]))

posts_to_categorize = list(posts_collection.find({"manualDefinition": {"$exists": False}}))

textsList = []
for idx, post in enumerate(posts_to_categorize):
    post_text = post.get('text', '')
    newline = '\n' if post_text else ''
    attached_post_text = post['attachedPost'].get('text', '') if post.get('attachedPost') else ''
    text = f"{post_text}{newline}{attached_post_text}"
    textsList.append(unidecode(text))

X = vectorizer.transform(textsList)


y_pred_proba = model.predict_proba(X)


all_predictions = []
for idx in range((len(posts_to_categorize))):
    prediction_dict = {category: prob for category, prob in zip(category_le.classes_, y_pred_proba[idx])}
    category = category_le.inverse_transform([np.argmax(y_pred_proba[idx])])[0]

    all_predictions.append({
        'text': textsList[idx],
        'predict': {
            'proba': prediction_dict,
            'category': category
        },
        'id': posts_to_categorize[idx]['id']
    })

SFH_predictions = [obj for obj in all_predictions if obj['predict']['category'] == 'SFH Deal']
confident_SFH_predictions = [obj for obj in all_predictions if obj['predict']['proba']['SFH Deal'] > 0.9]

print(len(SFH_predictions))
print(len(confident_SFH_predictions))

with open('./prompts/extract_SFH_info.txt') as file:
    extract_SFH_info_prompt = file.read()

completion = openai.chat.completions.create(
    messages=[
        {
            'role': 'system',
            'content': extract_SFH_info_prompt
        },
        {
            'role': 'user',
            'content': confident_SFH_predictions[0]['text']
        }
    ],
    model='gpt-4o',
    temperature=0,
    response_format={ "type": "json_object" }
)



print(f"POST ID: {confident_SFH_predictions[0]['id']}\n\n{confident_SFH_predictions[0]['text']}\n\n{confident_SFH_predictions[0]['predict']['proba']}\n\n{completion.choices[0].message.content}")