from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import re
import os
import json
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from unidecode import unidecode
from scipy.sparse import vstack
import matplotlib.pyplot as plt

from cuml.ensemble import RandomForestClassifier as cuRF

load_dotenv()


client = MongoClient(os.getenv('MONGODB_URI'))
posts_collection = client['JV-FINDER']['posts']


raw_labeled_posts = list(posts_collection.find({"manualDefinition": {"$exists": True}}))
formatted_labeled_posts = []
for idx, post in enumerate(raw_labeled_posts):
    post_text = post.get('text', '')
    newline = '\n' if post_text else ''
    attached_post_text = post['attachedPost'].get('text', '') if post.get('attachedPost') else ''
    text = f"{post_text}{newline}{attached_post_text}"
    formatted_labeled_posts.append({
        'index': idx,
        'text': unidecode(text),
        'category': post['manualDefinition']
    })


raw_unlabeled_posts = list(posts_collection.find({"manualDefinition": {"$exists": False}}))
formatted_unlabeled_posts = []
for idx, post in enumerate(raw_unlabeled_posts):
    post_text = post.get('text', '')
    newline = '\n' if post_text else ''
    attached_post_text = post['attachedPost'].get('text', '') if post.get('attachedPost') else ''
    text = f"{post_text}{newline}{attached_post_text}"
    formatted_unlabeled_posts.append({
        'index': idx + len(formatted_labeled_posts),  # continue the index from labeled posts
        'text': unidecode(text)
    })


df_labeled = pd.DataFrame(formatted_labeled_posts)
df_unlabeled = pd.DataFrame(formatted_unlabeled_posts)

category_le = LabelEncoder()
df_labeled['category'] = category_le.fit_transform(df_labeled['category'])
joblib.dump(category_le, 'label_encoder.joblib')

all_texts = pd.concat([df_labeled['text'], df_unlabeled['text']], ignore_index=True)

#Try other ngram_range
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4))
X_all = vectorizer.fit_transform(all_texts)
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')


X_labeled = X_all[:len(df_labeled)]
X_unlabeled = X_all[len(df_labeled):]
y_labeled = df_labeled['category'].values
y_unlabeled = -1 * np.ones(len(df_unlabeled))

X_combined = vstack([X_labeled, X_unlabeled])
y_combined = np.hstack((y_labeled, y_unlabeled))

X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.1, random_state=1)

base_model = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=42, n_jobs=-1)

self_training_model = SelfTrainingClassifier(base_model, criterion='k_best', k_best=10, max_iter=None)

self_training_model.fit(X_combined, y_combined)

joblib.dump(self_training_model, 'self_training_model.joblib')

y_pred = self_training_model.predict(X_test)

total_train_samples = X_combined.shape[0]
total_test_samples = y_test.shape[0]
correct_test_samples_count = np.sum(y_test == y_pred)

print(f"Train Samples: {total_train_samples}")
print(f"{correct_test_samples_count}/{total_test_samples} {correct_test_samples_count/total_test_samples:.2%}")
print(classification_report(y_test, y_pred, target_names=category_le.classes_))






# print(len(feature_names))

# plt.figure(figsize=(20,10))
# tree.plot_tree(model, filled=True, feature_names=feature_names, class_names=category_le.classes_)
# plt.savefig('plot.png', dpi=800)