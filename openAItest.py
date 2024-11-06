from openai import OpenAI
from openai import AsyncOpenAI
import asyncio
import configparser

# PUBLIC SITE 
openai_api_key = "" # FILL THIS IN
openai_api_base = "https://data.ai.uky.edu/llm-factory/openai/v1"
adapter_id = "" # FILL THIS IN

client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)



async def test():
    # synchrounous completions
    completion = await client.chat.completions.create(
        model=adapter_id,
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "model", "content": "Hello, what can I assist you with?"},
            {"role": "user", "content": "Hello, my name is john"},
            {"role": "model", "content": "Hello, John, what can I assist you with today?"},
            {"role": "user", "content": "What is my name?"},
        ],
        max_tokens = 200,
        temperature = 0.7,
    )
    print("Completion result:")
    print(completion)

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test())
