import os
import json
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
from unidecode import unidecode

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
posts_collection = client['JV-FINDER']['posts']

openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

with open('prompt.txt', 'r') as file:
    prompt = file.read()

posts = list(posts_collection.find({'manualDefinition': { '$exists': True }}))

for count, post in enumerate(posts):
    post_text = post.get('text', '')
    newline = '\n' if post_text else ''
    attached_post_text = post['attachedPost'].get('text', '') if post.get('attachedPost') else ''
    text = f"{post_text}{newline}{attached_post_text}"

    completion = openai.chat.completions.create(
        messages=[{
            'role': 'user',
            'content': f"{text}\n------\n{prompt}"
        }],
        model='gpt-4o',
        temperature=0,
        response_format={ "type": "json_object" }
    )

    response = json.loads(completion.choices[0].message.content)

    if response['result'] != post['manualDefinition']:
        print(json.dumps({
            'text': text,
            'manual_definition': post['manualDefinition'],
            'gpt_definition': response['result'],
            'gpt_reason': response['reason'],
            'post_id': post['id']
        }, indent=2))

        choice = input("Type 1 to Continue or Type 0 to End ")

        if choice == '0':
            break

    print(f"Gud: {count + 1}")

print('FINISHED')