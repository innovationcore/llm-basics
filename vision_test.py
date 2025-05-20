from openai import OpenAI
import configparser
import base64
import os

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

def encode_image(image_path):
    """
    Convert an image file to base64 encoding for API submission
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_image(image_path, prompt):
    """
    Send an image to a vision-capable model with a prompt
    
    Args:
        image_path: Path to the image file
        prompt: Text instructions for the model about the image
        
    Returns:
        Model's response about the image
    """
    # Check if image exists
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"
        
    # Encode the image as base64
    base64_image = encode_image(image_path)
    
    # Create the API request with image and prompt
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-90B-Vision-Instruct", # Select a multi-modal model
        messages=[
            {
                "role": "user",
                "content": [
                    # Text part of the message
                    {"type": "text", "text": prompt},
                    # Image part of the message
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    # Return the model's response
    return response.choices[0].message.content


def test():
    """Test the vision capabilities with a sample image"""
    
    # Path to your image file
    image_path = "sample_image.jpg"  # Replace with your image path
    
    # Example prompt for image analysis
    prompt = "What's in this image? Please describe it in detail."
    
    # Get and print the response
    result = analyze_image(image_path, prompt)
    print(f"Model's analysis of the image:\n{result}")


if __name__ == '__main__':
    test()