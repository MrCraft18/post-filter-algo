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
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn import tree
from unidecode import unidecode
from scipy.sparse import vstack
import matplotlib.pyplot as plt

load_dotenv()


client = MongoClient(os.getenv('MONGODB_URI'))
posts_collection = client['JV-FINDER']['posts']

print('Beginning')


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

print('Created Labeled and Unlabeled DataFrames')

category_le = LabelEncoder()
df_labeled['category'] = category_le.fit_transform(df_labeled['category'])
joblib.dump(category_le, 'label_encoder.joblib')

all_texts = pd.concat([df_labeled['text'], df_unlabeled['text']], ignore_index=True)


stop_words = list(ENGLISH_STOP_WORDS.union({'tx', 'worth', 'fort', 'dallas', 'texas', 'arlington', 'mckinney', 'irving', 'allen', 'joshua', 'sherman', 'prairie', 'corsicana', 'antonio', 'garland', 'carrollton', 'austin', 'greenville', 'frisco', 'denison', 'burleson' , 'forney', 'amarillo', 'waco', 'midlothian', 'saginaw', 'killeen', 'southlake', 'huntsville'}))

vectorizer = TfidfVectorizer(stop_words=stop_words, token_pattern=r"(?u)\b(?!\w*\d)\w+\b")
# vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4))
X_all = vectorizer.fit_transform(all_texts)
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("Created TD-IDF Features")
print(len(vectorizer.vocabulary_))


X_labeled = X_all[:len(df_labeled)]
X_unlabeled = X_all[len(df_labeled):]
y_labeled = df_labeled['category'].values
y_unlabeled = -1 * np.ones(len(df_unlabeled))

X_combined = vstack([X_labeled, X_unlabeled])
y_combined = np.hstack((y_labeled, y_unlabeled))

print('Combined Labeled and Unlabeled Datasets')

X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled, test_size=0.1, random_state=2)

base_model = RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=42, n_jobs=12)
# base_model = DecisionTreeClassifier(max_depth=10, random_state=42, max_features=None)

print('Training Start')

self_training_model = SelfTrainingClassifier(base_model, criterion='k_best', k_best=10, max_iter=50)

self_training_model.fit(X_combined, y_combined)

joblib.dump(self_training_model, 'self_training_model.joblib')

y_pred = self_training_model.predict(X_test)

y_pred_proba = np.array(self_training_model.predict_proba(X_test))
print(np.array(y_pred_proba))
misclassified_indices = np.where(y_test != y_pred)[0]
misclassified_proba = y_pred_proba[misclassified_indices]
print(misclassified_proba)


total_test_samples = y_test.shape[0]
correct_test_samples_count = np.sum(y_test == y_pred)

print(f"{correct_test_samples_count}/{total_test_samples} {correct_test_samples_count/total_test_samples:.2%}")
print(classification_report(y_test, y_pred, target_names=category_le.classes_))

# final_tree = self_training_model.base_estimator_

# feature_importances = final_tree.feature_importances_
# feature_names = vectorizer.get_feature_names_out()

# # Create a list of feature names with their corresponding importance scores
# important_features = [(feature_names[i], feature_importances[i]) for i in range(len(feature_importances))]

# # Sort the features by importance (optional)
# important_features.sort(key=lambda x: x[1], reverse=True)

# for feature, importance in important_features:
#     if (importance > 0):
#         print(f"Feature: {feature}, Importance: {importance}")


# feature_names = vectorizer.get_feature_names_out()

# plt.figure(figsize=(20, 10))
# tree.plot_tree(base_model, filled=True, feature_names=feature_names, class_names=category_le.classes_)
# plt.savefig('plot.png', dpi=800)