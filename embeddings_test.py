from openai import OpenAI
from openai import AsyncOpenAI
import asyncio
import configparser


# PUBLIC SITE 
openai_api_key = "" # FILL THIS IN 
openai_api_base = "https://data.ai.uky.edu/llm-factory/openai/v1"

client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

async def generate_embeddings(inputs, model):
    # Call the completion endpoint to generate embeddings
    response = await client.embeddings.create(input = inputs, model = model)
    print(response)
    embeddings = response
    return embeddings


if __name__ == '__main__':
    input_text = "Once upon a time, there was a king who ruled over a vast kingdom." # CHANGE THIS
    adapter_id = "" # FILL THIS IN 
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_embeddings(input_text, adapter_id))