
from dataclasses import asdict
import json

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation import Conversation
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider

class GroundTruthJudgeQuery(LLMQuery):

    def __init__(self, model_provider: ModelProvider, model_id: str, grading_rubric: str, assistant: Assistant, conversation: Conversation, user_persona: CharacterCard):
        super().__init__(model_provider, model_id)
        self.grading_rubric = grading_rubric
        self.assistant = assistant
        self.conversation = conversation
        self.user_persona = user_persona
        
    def generate_prompt(self):
        return f"""Predict the score that the specific user described below would assign to the AI assistant's performance in the given conversation. The user was provided with a specific grading rubric to follow when evaluating the assistant. Your task is to think from this particular user's perspective, considering their personality, expectations, and communication style.

### Considerations
- Analyze the conversation from the user's subjective viewpoint, not from an objective third-party perspective
- The user was given the provided grading rubric as their guide, but interpret it through their personal lens
- Consider how the user's personality, background, and goals would influence their satisfaction with the assistant's responses
- Factor in the user's communication style and what they would value most in an interaction
- Account for whether the assistant addressed the user's specific scenario and needs
- Consider the user's likely tolerance for different response styles, length, formality, etc.
- Think about what would frustrate or delight this particular user

### User
{json.dumps(asdict(self.user_persona), indent=4)}

### Assistant
{json.dumps(asdict(self.assistant), indent=4)}

### Grading Rubric (provided to the user for evaluation)
{self.grading_rubric}

### Conversation
{json.dumps(self.conversation.prompt_format, indent=4)}
"""

    def response_schema(self):
        properties = {
            "score": {
                "type": "number",
                "description": "Numerical score from 0-100 representing the predicted user evaluation of the assistant's performance"
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