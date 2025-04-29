from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import requests
import yaml
import os
import re

from synthetic_conversation_generation.data_models.conversation import Conversation, Message, ROLE

@dataclass
class InferenceEndpoint:
    url: str
    body: Dict[str, Any]
    headers: Dict[str, str]
    response_path: List[Union[str, int]]

    @classmethod
    def from_yaml(cls, schema_path: str):
        with open(schema_path, 'r') as f:
            schema_data = yaml.safe_load(f)
        
        # Process environment variables in the schema
        schema_data = cls._interpolate_env_vars(schema_data)

        return cls(
            url=schema_data['url'],
            body=schema_data['body'],
            headers=schema_data.get('headers', {}),
            response_path=schema_data['response_path']
        )
    
    @staticmethod
    def _interpolate_env_vars(data: Any) -> Any:
        """
        Recursively interpolate environment variables in the data structure.
        Environment variables should be in the format ${VAR_NAME}.
        """
        if isinstance(data, str):
            # Replace ${VAR_NAME} with the corresponding environment variable
            pattern = r'\${([A-Za-z0-9_]+)}'
            matches = re.findall(pattern, data)
            result = data
            for var_name in matches:
                env_value = os.environ.get(var_name)
                if env_value is None:
                    raise ValueError(f"Environment variable '{var_name}' not found")
                result = result.replace(f"${{{var_name}}}", env_value)
            return result
        elif isinstance(data, dict):
            return {k: InferenceEndpoint._interpolate_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [InferenceEndpoint._interpolate_env_vars(item) for item in data]
        else:
            return data
        
    def get_assistant_message(self, conversation: Conversation) -> Message:
        """
        Generate the next assistant message in the conversation by calling the inference endpoint.
        """
        # Prepare request payload
        payload = self.body.copy()
        payload['messages'] = [{'role': msg.role.name, 'content': msg.content} for msg in conversation.messages]
        
        # Make request to the inference endpoint
        response = requests.post(self.url, json=payload, headers=self.headers)
        response.raise_for_status()
        
        # Parse the response using the provided path
        response_data = response.json()
        result = response_data
        for key in self.response_path:
            result = result[key]
                
        # Create and return the assistant message
        return Message(
            role=ROLE.assistant,
            content=result,
            timestamp=datetime.now(),
            message_id=len(conversation.messages)
        )