from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import re
import os
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

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


def create_feature_list(post):
    feature_list = []
    for key_pattern in KeyCategoryPatterns:
        regex = re.compile(key_pattern['regEx'], re.IGNORECASE)
        feature_list.append(int(bool(regex.search(post['text']))))
    return feature_list

category_le = LabelEncoder()

X = pd.DataFrame([create_feature_list(post) for post in formatted_training_posts])
y = pd.DataFrame(category_le.fit_transform([post['category'] for post in formatted_training_posts]))

indices = np.arange(len(X))
X['index'] = indices
y['index'] = indices

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_features = X_train.copy()

train_indices = X_train.pop('index')
test_indices = X_test.pop('index')
y_train = y_train.drop(columns=['index'])
y_test = y_test.drop(columns=['index'])

model = KNeighborsClassifier(n_neighbors=13, algorithm='kd_tree')
model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)
total_train_samples = len(X_train)
total_test_samples = len(y_test)
correct_test_samples_count = np.sum(np.array(y_test.squeeze()) == np.array(y_pred))

print(f"Train Samples: {total_train_samples}")
print(f"{correct_test_samples_count}/{total_test_samples} {correct_test_samples_count/total_test_samples:.2%}")
print(classification_report(y_test, y_pred, target_names=category_le.classes_))



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
        'features': X.loc[X['index'] == idx].drop(columns=['index']).values[0].tolist()
    })

# print(json.dumps(incorrect_classifications[0],indent=2))







# feature_names = [(key_pattern['regEx']) for key_pattern in KeyCategoryPatterns]

# plt.figure(figsize=(20,10))
# tree.plot_tree(model, filled=True, feature_names=feature_names, class_names=category_le.classes_)
# plt.savefig('plot.png', dpi=800)