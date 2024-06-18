import os
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
from unidecode import unidecode

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
posts_collection = client['JV-FINDER']['posts']

openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

print(os.getenv('OPENAI_API_KEY'))


completion = openai.chat.completions.create(
    messages=[{
        'role': 'user',
        'content': "Hello!"
    }],
    model='gpt-4o',
    temperature=0,
    # response_format={ "type": "json_object" }
)

print(completion)