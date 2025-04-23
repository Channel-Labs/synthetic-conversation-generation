from typing import List

from data_models.assistant import Assistant
from data_models.character_card import CharacterCard
from llm_queries.llm_query import LLMQuery, ModelProvider

class UserPersonaGenerator(LLMQuery):

    def __init__(self, model_provider: ModelProvider, model_id: str, assistant: Assistant, num_personas: int):
        super().__init__(model_provider, model_id)
        self.assistant = assistant
        self.num_personas = num_personas

    def generate_prompt(self):
        return f"""Generate {self.num_personas} users that are likely to interact with the assistant. The generated usersshould be as distinct and diverse as possible, in order to cover a wide range of user personas.

### Assistant
{self.assistant}
"""
    
    def response_schema(self):
        properties = {}
        for i in range(self.num_personas):
            properties[f"user_persona_{i}"] = {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The user's name"
                    },
                    "description": {
                        "type": "string",
                        "description": "An overview of the user's physical and mental traits."
                    },
                    "personality": {
                        "type": "string",
                        "description": "A description of the user's personality."
                    },
                    "scenario": {
                        "type": "string",
                        "description": "The context and circumstances in which the user enters the conversation."      
                    }
                },
                "required": ["name", "description", "personality", "scenario"],
                "additionalProperties": False
            }

        return {
            "type": "object",
            "properties": properties,
            "required": [f"user_persona_{i}" for i in range(self.num_personas)],
            "additionalProperties": False
        }
    
    def parse_response(self, json_response) -> List[CharacterCard]:   
        user_personas = []
        for i in range(self.num_personas):
            persona_json = json_response[f"user_persona_{i}"]
            user_persona = CharacterCard(
                name=persona_json["name"],
                description=persona_json["description"],
                personality=persona_json["personality"],
                scenario=persona_json["scenario"]
            )
                        
            user_personas.append(user_persona)
        
        return user_personas
