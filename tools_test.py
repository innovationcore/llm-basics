from openai import OpenAI
import configparser
import json

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

class Tool:
    '''
    A base class for tools with descriptions and functionality.
    '''
    def __init__(self, name, description, parameters):
        '''
        Initialize the tool with a name, description, and parameters schema.
        '''
        self.name = name
        self.description = description
        self.parameters = parameters

    def execute(self, **kwargs):
        '''
        Execute the tool's functionality. This should be overridden by subclasses.
        '''
        raise NotImplementedError('Subclasses must override the execute method.')


class CalculatorTool(Tool):
    '''
    A tool for performing basic arithmetic operations.
    '''
    def __init__(self):
        super().__init__(
            name='calculator',
            description='Performs basic arithmetic operations like addition, subtraction, multiplication, and division.',
            parameters={
                'operation': {
                    'description': 'The arithmetic operation to perform (e.g., addition, subtraction).',
                    'type': 'string'
                },
                'values': {
                    'description': 'A list of numbers to perform the operation on.',
                    'type': 'list'
                }
            }
        )

    def execute(self, operation, values):
        '''
        Perform the specified arithmetic operation on the given values.
        '''
        if operation == 'addition':
            return sum(values)
        elif operation == 'subtraction':
            return values[0] - sum(values[1:])
        elif operation == 'multiplication':
            result = 1
            for value in values:
                result *= value
            return result
        elif operation == 'division':
            result = values[0]
            for value in values[1:]:
                result /= value
            return result
        else:
            raise ValueError(f'Unsupported operation: {operation}')


class WeatherLookupTool(Tool):
    '''
    A tool for looking up the weather in a specified city.
    '''
    def __init__(self):
        super().__init__(
            name='weather_lookup',
            description='Provides current weather information for a given city.',
            parameters={
                'city': {
                    'description': 'The name of the city to check the weather for.',
                    'type': 'string'
                }
            }
        )

    def execute(self, city):
        '''
        Mock implementation of weather lookup.
        '''
        # Replace with actual weather API call in production
        return f'The weather in {city} is sunny with a temperature of 25°C.'


# Create tool instances
calculator_tool = CalculatorTool()
weather_lookup_tool = WeatherLookupTool()

# Make tools available in a dictionary
tools = {
    calculator_tool.name: calculator_tool,
    weather_lookup_tool.name: weather_lookup_tool
}

# Format tool descriptions for the OpenAI API
tool_descriptions = [
    {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description,
            'parameters': {
                'type': 'object',
                'properties': {
                    key: {
                        'type': value['type'],
                        'description': value['description']
                    } for key, value in tool.parameters.items()
                },
                'required': list(tool.parameters.keys())
            }
        }
    }
    for tool_name, tool in tools.items()
]


def handle_function_call(function_call):
    """
    Execute the appropriate tool based on the function call from the model
    """
    tool_name = function_call.name
    arguments = json.loads(function_call.arguments)
    
    try:
        called_tool = tools[tool_name]
        return called_tool.execute(**arguments)
    except Exception as e:
        print(f"Error executing tool: {e}")
        return f"Error: {str(e)}"


def test():
    """Test tool usage with a simple math question"""
    
    # Set up the conversation with system and user messages
    messages = [
        {
            'role': 'system',
            'content': (
                'You are a helpful assistant. You can use the following tools:\n' +
                '\n'.join([f'{tool["function"]["name"]}: {tool["function"]["description"]}' for tool in tool_descriptions])
            )
        },
        {
            'role': 'user',
            'content': 'What is 15183172 + 1495186379?'
        }
    ]

    # Get initial model response with available tools
    completion = client.chat.completions.create(
        model='/models/functionary-small-v2.5',
        messages=messages, 
        max_tokens=200,
        temperature=0.7,
        tools=tool_descriptions  # Make tools available to the model
    )

    # Handle the assistant's response
    assistant_response = completion.choices[0].message
    
    # Check if the model wants to use a tool
    if assistant_response.tool_calls:
        print("Assistant wants to use tools:", assistant_response)
        
        # Process each tool call
        for tool_call in assistant_response.tool_calls:
            # Execute the function
            result = handle_function_call(tool_call.function)

            # Send the result back to the assistant
            messages.append(assistant_response)
            messages.append({
                'role': 'tool',
                'content': str(result),
                'tool_call_id': tool_call.id
            })

            # Print conversation history
            for message in messages:
                print(message)

            # Get final response after tool use
            completion = client.chat.completions.create(
                model='/models/functionary-small-v2.5',
                messages=messages, 
                max_tokens=200,
                temperature=0.7
            )
            
            print("Final response:")
            print(completion.choices[0].message)
    else:
        print('Assistant\'s direct response:')
        print(assistant_response)


if __name__ == '__main__':
    test()