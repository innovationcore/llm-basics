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

def test():
    """
    Transcribe an audio file to text using the Whisper model
    """
    # Open the audio file in binary read mode
    audio_file = open("fitnessgram.mp3", "rb")
    
    # Call the transcriptions endpoint to convert speech to text
    transcription = client.audio.transcriptions.create(
        model="whisper-1",  # Using Whisper model for transcription
        file=audio_file,
    )
    
    # Print the transcription result
    print("Transcription result:")
    print(transcription)

if __name__ == '__main__':
    test()