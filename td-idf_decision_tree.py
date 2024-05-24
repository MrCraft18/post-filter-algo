from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import re
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
import matplotlib.pyplot as plt

from cuml.ensemble import RandomForestClassifier as cuRF

from KeyCategoryPatterns import KeyCategoryPatterns

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
posts_collection = client['JV-FINDER']['posts']

raw_posts = list(posts_collection.find({"manualDefinition": {"$exists": True}}))

formatted_training_posts = []
for idx, post in enumerate(raw_posts):
    post_text = post.get('text', '')
    newline = '\n' if post_text else ''
    attached_post_text = post['attachedPost'].get('text', '') if post.get('attachedPost') else ''
    text = f"{post_text}{newline}{attached_post_text}"
    formatted_training_posts.append({
        'index': idx,
        'text': text,
        'category': post['manualDefinition']
    })

df = pd.DataFrame(formatted_training_posts)

category_le = LabelEncoder()
df['category'] = category_le.fit_transform(df['category'])

# vectorizer = TfidfVectorizer(stop_words='english', token_pattern=r"(?u)\b(?!\w*\d)\w+\b")
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,4))
X = vectorizer.fit_transform(df['text']).astype(np.float32).toarray()
y = df['category']


indices = np.arange(X.shape[0])
df['index'] = indices

X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.1, random_state=42)

model = cuRF(n_estimators=2000, max_depth=1000, random_state=42, max_features=1.0, n_streams=2)
# model = RandomForestClassifier(n_estimators=600, max_depth=None, random_state=42, max_features=None)
# model = DecisionTreeClassifier(max_depth=None, random_state=42, max_features=None)
# model = KNeighborsClassifier(n_neighbors=101, algorithm='auto')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


total_train_samples = X_train.shape[0]
total_test_samples = y_test.shape[0]
correct_test_samples_count = np.sum(y_test == y_pred)

print(f"Train Samples: {total_train_samples}")
print(f"{correct_test_samples_count}/{total_test_samples} {correct_test_samples_count/total_test_samples:.2%}")
print(classification_report(y_test, y_pred, target_names=category_le.classes_))

feature_names = vectorizer.get_feature_names_out()

importances = model.feature_importances_
used_feature_indices = np.nonzero(importances)[0]
used_feature_names = [feature_names[i] for i in used_feature_indices]
print(used_feature_names)



incorrect_mask = y_test.squeeze() != y_pred
incorrect_indices = test_indices[incorrect_mask]
incorrect_predictions = y_pred[incorrect_mask]

incorrect_df = pd.DataFrame({
    'index': incorrect_indices,
    'incorrect_prediction': incorrect_predictions
})

incorrect_classifications = []
for idx in incorrect_df['index']:
    post = formatted_training_posts[idx]
    incorrect_classifications.append({
        'text': post['text'],
        'category': post['category'],
        'filter_assigned_category': category_le.inverse_transform([incorrect_df.loc[incorrect_df['index'] == idx, 'incorrect_prediction'].values[0]])[0],
        'features': X[idx].toarray()[0].tolist()
    })

# print(json.dumps(incorrect_classifications[0],indent=2))








# print(len(feature_names))

# plt.figure(figsize=(20,10))
# tree.plot_tree(model, filled=True, feature_names=feature_names, class_names=category_le.classes_)
# plt.savefig('plot.png', dpi=800)