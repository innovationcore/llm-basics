from openai import OpenAI
import configparser

# Load configuration from config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_base = config.get('API', 'openai_api_base')
openai_api_key = config.get('API', 'openai_api_key')
adapter_id = ""  # Add your LLM Factory adapter ID here

# Initialize OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def test():
    """Send a test conversation to the chat model and print the response"""
    
    # Create a chat completion with predefined messages
    completion = client.chat.completions.create(
        model=adapter_id,
        messages=[
            # System message defines the AI assistant's behavior
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            # Simulated conversation history
            {"role": "model", "content": "Hello, what can I assist you with?"},
            {"role": "user", "content": "Hello, my name is john"},
            {"role": "model", "content": "Hello, John, what can I assist you with today?"},
            # User's current query
            {"role": "user", "content": "What is my name?"},
        ],
        max_tokens=200,     # Maximum length of the response in tokens
        temperature=0.7,    # Controls randomness (0=deterministic, 1=creative)
    )
    
    # Print the full response
    print("Completion result:")
    print(completion)
    return 

if __name__ == '__main__':
    test()