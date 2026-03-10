from openai import OpenAI
from dotenv import load_dotenv
import os

# 1. 加载环境变量
load_dotenv()

# 2. 创建OpenAI客户端 （指向 DeepSeek)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)

# 3. 调用 Chat API
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "你好，请用一句话介绍RAG技术"}
    ]
)

print(response.choices[0].message.content)