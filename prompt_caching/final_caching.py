import anthropic
import os
import time
import json
from dotenv import load_dotenv
from metaprompt import meta_prompt, hash_mapping
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
MODEL_NAME = "claude-3-5-sonnet-20240620"

# Load the document
def load_data(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

document_data = load_data("/home/chandan/Projects/prompt_caching/data/2021.03.31_TI SMRS_PCAP - Atomic III.pdf")

def get_json_from_response(response):
    lines = response.strip().split('\n')
    data = {}
    for line in lines:
        key, value = line.split('=', 1)
        data[key] = value
    return data

def remap_keys(data, hash_mapping):
    remapped_data = {}
    for new_key, old_key in hash_mapping.items():
        if old_key in data:
            remapped_data[new_key] = data[old_key]
    return remapped_data

class ConversationHistory:
    def __init__(self):
        self.turns = []

    def add_turn_assistant(self, content):
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
        result = []
        user_turns_processed = 0
        for turn in reversed(self.turns):
            if turn["role"] == "user" and user_turns_processed < 2:
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
                result.append(turn)
        return list(reversed(result))

# Initialize the conversation history
conversation_history = ConversationHistory()

# Predefined question for extracting data
question = ["""Respond with a list of key-value pairs, 
       one per line, ensuring EVERY SINGLE critical data point 
       from the template is captured for EACH investor found in the 
       document. You must strictly make sure that you follow the 
       key-mapping mentioned to you as a part of your system prompt.""",
       """Respond with a list of key-value pairs, 
       one per line, ensuring EVERY SINGLE critical data point 
       from the template is captured for EACH investor found in the 
       document. You must strictly make sure that you follow the 
       key-mapping mentioned to you as a part of your system prompt."""]

def simulate_conversation_for_all_investors(docs, hash_mapping):
    final_responses = []
    
    # Loop through each page in the document
    for i, doc in enumerate(docs, 1):
        print(f"\nProcessing Investor {i}:")
        
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
                    "text": f"Here is the parsed JSON data from a financial PDF document to analyze:\n{doc.page_content}"
                }
            ],
            messages=conversation_history.get_turns(),
        )

        # Extract the assistant's reply
        assistant_reply = response.content[0].text
        print(f"Assistant: {assistant_reply}")
        
        # Convert to JSON and remap keys
        json_data = get_json_from_response(assistant_reply)
        remapped_data = remap_keys(json_data, hash_mapping)
        final_responses.append(remapped_data)

        # Add assistant's reply to conversation history
        conversation_history.add_turn_assistant(assistant_reply)

        # Record the end time
        end_time = time.time()

        # Print elapsed time
        elapsed_time = end_time - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")

    return final_responses

# Run the conversation simulation for all investors
investor_data = simulate_conversation_for_all_investors(document_data, hash_mapping)

# Save the final JSON list to a file
with open('investors_data.json', 'w') as f:
    json.dump(investor_data, f, indent=4)

print("All investors' data extracted and saved to 'investors_data.json'")
