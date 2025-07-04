
from dataclasses import asdict
from datetime import datetime
import json

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.conversation import Conversation, Message, ROLE
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider

class UserMessageQuery(LLMQuery):

    def __init__(self, model_provider: ModelProvider, model_id: str, conversation: Conversation, user_persona: CharacterCard, assistant: Assistant):
        super().__init__(model_provider, model_id)
        self.conversation = conversation
        self.user_persona = user_persona
        self.assistant = assistant
        
    def generate_prompt(self):
        return f"""Generate a realistic, conversational user response that would naturally follow next in this dialogue between a human user and an AI assistant.

### Instructions
- Authentically reflect the user's defined personality, background, and communication style
- Use natural human speech patterns (varied sentence length, occasional grammatical imperfections, use of contractions, etc.)
- Avoid overusing the assistant's name (humans rarely address others by name in every message)
- Include appropriate human emotions, hesitations, or thought processes based on the conversation context
- Mimic human behavior when chatting with AI (generally concise, direct questions, sometimes abrupt topic changes, occasional follow-ups without pleasantries, varying engagement depth based on interest level, etc.)

### User Definition
{json.dumps(asdict(self.user_persona), indent=4)}

### Assistant Definition
{json.dumps(asdict(self.assistant), indent=4)}

### Conversation History
{json.dumps(self.conversation.prompt_format, indent=4)}
"""

    def response_schema(self):
        properties = {}
        properties["user_message"] = {
            "type": "string",
            "description": "The user's next message in the conversation"
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
