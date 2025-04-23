
from datetime import datetime

from data_models.assistant import Assistant
from data_models.conversation import Conversation, Message, ROLE
from data_models.user_persona import UserPersona
from llm_queries.llm_query import LLMQuery, ModelProvider

class UserMessageGenerator(LLMQuery):

    def __init__(self, model_provider: ModelProvider, model_id: str, conversation: Conversation, user_persona: UserPersona, assistant: Assistant):
        super().__init__(model_provider, model_id)
        self.conversation = conversation
        self.user_persona = user_persona
        self.assistant = assistant
    def generate_prompt(self):
        return f"""Generate the next user message in the conversation between the user and the assistant.

### User Definition
{self.user_persona}

### Assistant Definition
{self.assistant}

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
    
    def parse_response(self, json_response) -> Message:   
        return Message(
            message_id=len(self.conversation.messages),
            role=ROLE.user,
            content=json_response["user_message"],
            timestamp=datetime.now()
        )
