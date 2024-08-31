import anthropic
import os
import time
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
from metaprompt import meta_prompt, hash_mapping
from langchain_community.document_loaders import PyPDFLoader

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
MODEL_NAME = "claude-3-5-sonnet-20240620"

def load_data(s3_path):
    # loader = S3FileLoader(
    # "testing-hwc", "fake.docx", aws_access_key_id="xxxx", aws_secret_access_key="yyyy")
    loader = PyPDFLoader(s3_path)
    docs = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # splits = text_splitter.split_documents(docs)
    # vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    # retriever = vectorstore.as_retriever()
    return docs

document_data = load_data("/home/chandan/Projects/prompt_caching/data/2021.03.31_TI SMRS_PCAP - Atomic III.pdf")


import json
def get_json_from_response(response):
    # text = response.content[0].text
    # Split the content by lines and then by the equals sign to create key-value pairs
    lines = response.strip().split('\n')
    data = {}
    for line in lines:
        key, value = line.split('=', 1)
        data[key] = value
    # Convert to JSON format
    json_data = json.dumps(data, indent=4)
    # Print the JSON formatted string
    print(json_data)
    return json_data

def remap_keys(data, hash_mapping):
    remapped_data = {}
    for new_key, old_key in hash_mapping.items():
        if old_key in data:
            remapped_data[new_key] = data[old_key]
    return remapped_data

class ConversationHistory:
    def __init__(self):
        # Initialize an empty list to store conversation turns
        self.turns = []

    def add_turn_assistant(self, content):
        # Add an assistant's turn to the conversation history
        self.turns.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })

    def add_turn_user(self, content):
        # Add a user's turn to the conversation history
        self.turns.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": content
                }
            ]
        })

    def get_turns(self):
        # Retrieve conversation turns with specific formatting
        result = []
        user_turns_processed = 0
        # Iterate through turns in reverse order
        for turn in reversed(self.turns):
            if turn["role"] == "user" and user_turns_processed < 2:
                # Add the last two user turns with ephemeral cache control
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": turn["content"][0]["text"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                })
                user_turns_processed += 1
            else:
                # Add other turns as they are
                result.append(turn)
        # Return the turns in the original order
        return list(reversed(result))

# Initialize the conversation history
conversation_history = ConversationHistory()

# Predefined questions for our simulation
questions = [
    """Respond with a list of key-value pairs, 
       one per line, ensuring EVERY SINGLE critical data point 
       from the template is captured for EACH investor found in the 
       document. You must strictly make sure that you follow the 
       key-mapping mentioned to you as a part of your system prompt""",
    """Respond with a list of key-value pairs, 
       one per line, ensuring EVERY SINGLE critical data point 
       from the template is captured for EACH investor found in the 
       document. You must strictly make sure that you follow the 
       key-mapping mentioned to you as a part of your system prompt""",
    """Respond with a list of key-value pairs, 
       one per line, ensuring EVERY SINGLE critical data point 
       from the template is captured for EACH investor found in the 
       document. You must strictly make sure that you follow the 
       key-mapping mentioned to you as a part of your system prompt""",
]

def simulate_conversation():
    final_responses = []
    for i, question in enumerate(questions, 1):
        print(f"\nTurn {i}:")
        print(f"User: {question}")
        
        # Add user input to conversation history
        conversation_history.add_turn_user(question)

        # Record the start time for performance measurement
        start_time = time.time()

        # Make an API call to the assistant
        response = client.messages.create(
            model=MODEL_NAME,
            extra_headers={
              "anthropic-beta": "prompt-caching-2024-07-31"
            },
            max_tokens=4096,
            system=[
                    {
                        "type": "text",
                        "text": meta_prompt
                    },
                    {
                        "type": "text",
                        "text": f"""Here is the parsed JSON data from a financial PDF document to analyze:        
                                {document_data}
                                """,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
            messages=conversation_history.get_turns(),
        )

        # Record the end time
        end_time = time.time()

        # Extract the assistant's reply
        assistant_reply = response.content[0].text
        # print(f"Assistant: {assistant_reply}")
        # # Convert to JSON and remap keys
        # json_data = json.loads(get_json_from_response(assistant_reply))
        # remapped_data = remap_keys(json_data, hash_mapping)
        final_responses.append(assistant_reply)

        # Print token usage information
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        input_tokens_cache_read = getattr(response.usage, 'cache_read_input_tokens', '---')
        input_tokens_cache_create = getattr(response.usage, 'cache_creation_input_tokens', '---')
        print(f"User input tokens: {input_tokens}")
        print(f"Output tokens: {output_tokens}")
        print(f"Input tokens (cache read): {input_tokens_cache_read}")
        print(f"Input tokens (cache write): {input_tokens_cache_create}")

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time

        # Calculate the percentage of input prompt cached
        total_input_tokens = input_tokens + (int(input_tokens_cache_read) if input_tokens_cache_read != '---' else 0)
        percentage_cached = (int(input_tokens_cache_read) / total_input_tokens * 100 if input_tokens_cache_read != '---' and total_input_tokens > 0 else 0)

        print(f"{percentage_cached:.1f}% of input prompt cached ({total_input_tokens} tokens)")
        print(f"Time taken: {elapsed_time:.2f} seconds")

        # Add assistant's reply to conversation history
        conversation_history.add_turn_assistant(assistant_reply)
        print(f"final responses: {final_responses}")
    return final_responses

# ans = [] 
# Run the simulated conversation
ans = simulate_conversation()
len(ans)
# import json
# json_data = get_json_from_response(ans[0]))
# print(json_data)

# type(json_data)

# final_response = remap_keys(json_data, hash_mapping)

def remove_redundant_text(data_list):
    unique_list = []
    accumulated_text = ""

    for idx, item in enumerate(data_list):
        # Remove accumulated_text from the start of the current item if present
        if accumulated_text and item.startswith(accumulated_text):
            unique_part = item[len(accumulated_text):]
        else:
            unique_part = item
        
        unique_list.append(unique_part.strip())
        accumulated_text += unique_part

    return unique_list

# Process the list
clean_list = remove_redundant_text(ans)
final_ans = []
for ele in clean_list:
    json_data = get_json_from_response(ele)
    final_ans.append(json_data)

result = []
for json_ele in final_ans:
    remapped_data = remap_keys(json.loads(json_ele), hash_mapping)
    result.append(remapped_data)

from prettyprint import print
print(result)