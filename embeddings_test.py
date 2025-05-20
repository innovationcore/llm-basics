from openai import OpenAI
import configparser

# Load configuration from config.ini file
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_base = config.get('API', 'openai_api_base')
openai_api_key = config.get('API', 'openai_api_key')

# Initialize OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def generate_embeddings(inputs):
    """
    Generate vector embeddings for the given text input
    
    Args:
        inputs: Text to create embeddings for
        model: LLM Factory adapter ID for embeddings
        
    Returns:
        Vector embedding representation of the input text
    """
    # Call the embeddings endpoint to generate vector representations
    response = client.embeddings.create(input=inputs, model="")
    print(response)
    return response


if __name__ == '__main__':
    # Text to be converted into vector embeddings
    input_text = "Once upon a time, there was a king who ruled over a vast kingdom."
    
    generate_embeddings(input_text)