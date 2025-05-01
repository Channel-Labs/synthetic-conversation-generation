from __future__ import annotations

from abc import ABC, abstractmethod
import json
import time
import logging
from typing import Dict

import anthropic
import openai


logger = logging.getLogger(__name__)


class LLMQuery(ABC):
    """Base abstract class for LLM queries that defines the common interface."""
    
    @abstractmethod
    def __init__(self, model_provider: ModelProvider, model_id: str):
        self.model_provider = model_provider
        self.model_id = model_id
    
    @abstractmethod
    def generate_prompt(self) -> str:
        """Generate the prompt to send to the LLM."""
        pass
    
    @abstractmethod
    def response_schema(self):
        """Define the expected response schema."""
        pass
    
    @abstractmethod
    def parse_response(self, json_response):
        """Parse the JSON response from the LLM."""
        pass
    
    def query(self, max_retries=3, retry_delay=2, timeout=60):
        """Send the query to the LLM and return the parsed response."""
        user_msg = self.generate_prompt()
        response_schema = self.response_schema()

        retries = 0
        while retries < max_retries:
            try:
                response = self.model_provider.query(user_msg, response_schema, self.model_id, timeout)
                return self.parse_response(response)
            except Exception as e:
                retries += 1
                logger.error(f"Error: {e}")
                logger.info(f"Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay += 2

        raise Exception("Unable to complete llm query.")

    
class ModelProvider(ABC):

    @abstractmethod
    def query(self, user_msg: str, response_schema: Dict, model_id: str, timeout: int=60):
        pass

    @abstractmethod
    def response_format(self, response_schema: Dict) -> Dict:
        pass

class OpenAIModelProvider(ModelProvider):

    def __init__(self, client: openai.OpenAI):
        self.client = client

    def query(self, user_msg: str, response_schema: Dict, model_id: str, timeout: int=60):       
        response = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": user_msg}
            ],
            seed=42,
            response_format=self.response_format(response_schema),
            timeout=timeout
        ).choices[0].message.content

        return json.loads(response)
    
    def response_format(self, response_schema: Dict):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": response_schema
            }
        }

class AnthropicModelProvider(ModelProvider):
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
    
    def query(self, user_msg: str, response_schema: Dict, model_id: str, timeout: int=60):
        """Handle API calls to Anthropic Claude using the tools API for schema enforcement"""
        response_format = self.response_format(response_schema)
        
        response = self.client.messages.create(
            model=model_id,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_msg}
            ],
            tools=[response_format],
            tool_choice={"type": "tool", "name": response_format["name"]},
            timeout=timeout
        )
        
        # Parse the response to get the tool use
        for content in response.content:
            if content.type == "tool_use":
                return content.input
        
        # Fallback in case the model didn't use the tool
        raise Exception("Anthropic model did not return a tool use response")
    
    def response_format(self, response_schema: Dict) -> Dict:
        return {
            "name": "json_extractor",
            "description": "Extract structured data according to the provided schema",
            "input_schema": response_schema
        }