
from dataclasses import asdict
import json

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.conversation import Conversation
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider

class JudgeConversationQuery(LLMQuery):

    def __init__(self, model_provider: ModelProvider, model_id: str, grading_rubric: str, assistant: Assistant, conversation: Conversation):
        super().__init__(model_provider, model_id)
        self.grading_rubric = grading_rubric
        self.assistant = assistant
        self.conversation = conversation
        
    def generate_prompt(self):
        return f"""Assess the assistant's performance in the following conversation on a scale of 0 to 100, based on the grading rubric provided.

### Grading Rubric
{self.grading_rubric}

### Assistant
{json.dumps(asdict(self.assistant), indent=4)}

### Conversation
{json.dumps(self.conversation.prompt_format, indent=4)}"""

    def response_schema(self):
        properties = {
            "score": {
                "type": "number",
                "description": "A score between 0 and 100, representing how well the assistant performed in the conversation"
            }
        }

        return {
            "type": "object",
            "properties": properties,
            "required": ["score"],
            "additionalProperties": False
        }
    
    def parse_response(self, json_response) -> int:   
        return int(json_response["score"])