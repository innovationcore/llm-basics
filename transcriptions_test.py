from openai import OpenAI
from openai import AsyncOpenAI
import asyncio
import configparser

# PUBLIC SITE 
openai_api_base = "https://data.ai.uky.edu/llm-factory/openai/v1"
openai_api_key = ""

client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

async def test():
    # synchrounous completions
    audio_file= open("fitnessgram.mp3", "rb")
    completion = await client.audio.transcriptions.create(
        model = "whisper-1",
        file = audio_file,
    )
    print("Transcription result:")
    print(completion)

if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(test())
