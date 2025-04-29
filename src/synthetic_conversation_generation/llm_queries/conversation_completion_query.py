from dataclasses import asdict
import json

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.conversation import Conversation
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider

class ConversationCompletionQuery(LLMQuery):

    def __init__(self, model_provider: ModelProvider, model_id: str, conversation: Conversation,
                 user_persona: CharacterCard, assistant: Assistant):
        super().__init__(model_provider, model_id)
        self.conversation = conversation
        self.user_persona = user_persona
        self.assistant = assistant

    def generate_prompt(self):
        return f"""Determine whether the conversation below between a human user and an AI assistant has concluded.

### Considerations
- Has the primary user need or question been addressed satisfactorily?
- Is the user frustrated or has the conversation reached a dead end?
- Are there closure signals like gratitude, goodbyes, or acknowledgment of completion?
- Does the conversation feel complete based on natural human conversation patterns?
- Would a typical user naturally respond again or has the conversation concluded?

### User Definition
{json.dumps(asdict(self.user_persona), indent=4)}

### Assistant Definition
{json.dumps(asdict(self.assistant), indent=4)}

### Conversation
{json.dumps(self.conversation.prompt_format, indent=4)}
"""

    def response_schema(self):
        return {
            "type": "object",
            "properties": {
                "is_complete": {
                    "type": "boolean",
                    "description": "True if the conversation has reached a natural conclusion, False if it should continue"
                }
            },
            "required": ["is_complete"],
            "additionalProperties": False
        }

    def parse_response(self, json_response):
        return json_response["is_complete"]