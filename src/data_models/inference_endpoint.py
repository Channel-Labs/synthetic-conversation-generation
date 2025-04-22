from dataclasses import dataclass
from typing import Dict, Any
import requests
import yaml

from data_models.conversation import Conversation, Message, ROLE

@dataclass
class InferenceEndpoint:
    url: str
    params: Dict[str, Any]
    response_parser: str

    @classmethod
    def from_yaml(cls, schema_path: str):
        with open(schema_path, 'r') as f:
            schema_data = yaml.safe_load(f)

        return cls(
            url=schema_data['url'],
            params=schema_data['params'],
            response_parser=schema_data['response_parser']
        )
        
    def get_assistant_message(self, conversation: Conversation) -> Message:
        """
        Generate the next assistant message in the conversation by calling the inference endpoint.
        """
        # Prepare request payload
        payload = self.params.copy()
        payload['messages'] = [{'role': msg.role.value, 'content': msg.content} for msg in conversation.messages]
        
        # Make request to the inference endpoint
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        
        # Parse the response based on the configured parser
        response_data = response.json()
        if self.response_parser == 'openai':
            assistant_response = response_data['choices'][0]['message']['content']
        elif self.response_parser == 'anthropic':
            assistant_response = response_data['content'][0]['text']
        else:
            assistant_response = response_data.get('content', response_data.get('output', ''))
        
        # Create and return the assistant message
        return Message(
            role=ROLE.assistant,
            content=assistant_response
        )