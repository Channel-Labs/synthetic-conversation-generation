from data_models.conversation import Conversation
from llm_queries.llm_query import LLMQuery, ModelProvider

class UserMessageGenerator(LLMQuery):

    def __init__(self, model_provider: ModelProvider, model_id: str, conversation: Conversation):
        super().__init__(model_provider, model_id)
        self.conversation = conversation

    def generate_prompt(self):
        return f"""Generate the next user message in the conversation.

### Conversation History
{self.conversation}
"""
    
    def response_schema(self):
        properties = {}
        properties["user_message"] = {
            "type": "string",
            "description": "The next user message in the conversation"
        }

        return {
            "type": "object",
            "properties": properties,
            "required": ["user_message"],
            "additionalProperties": False
        }
    
    def parse_response(self, json_response):   
        return json_response["user_message"]
