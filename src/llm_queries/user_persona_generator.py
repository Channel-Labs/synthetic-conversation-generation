from typing import List

from data_models.assistant import Assistant
from data_models.user_persona import UserPersona, NumericAttribute, TextAttribute
from llm_queries.llm_query import LLMQuery, ModelProvider

class UserPersonaGenerator(LLMQuery):

    def __init__(self, model_provider: ModelProvider, model_id: str, assistant: Assistant, num_personas: int):
        super().__init__(model_provider, model_id)
        self.assistant = assistant
        self.num_personas = num_personas

    def generate_prompt(self):
        return f"""Generate {self.num_personas} user personas that are likely to interact with the assistant. The personas should be distinct and diverse, in order to cover a range of user types.

### Assistant
{self.assistant}
"""
    
    def response_schema(self):
        properties = {}
        for i in range(self.num_personas):
            properties[f"user_persona_{i}"] = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "overview": {"type": "string"},
                    "numeric_attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "number"},
                                "description": {"type": "string"},
                                "min_value": {"type": "number"},
                                "max_value": {"type": "number"}
                            },
                            "required": ["name", "value", "description", "min_value", "max_value"],
                            "additionalProperties": False
                        }
                    },
                    "text_attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["name", "value", "description"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["name", "overview", "numeric_attributes", "text_attributes"],
                "additionalProperties": False
            }

        return {
            "type": "object",
            "properties": properties,
            "required": [f"user_persona_{i}" for i in range(self.num_personas)],
            "additionalProperties": False
        }
    
    def parse_response(self, json_response) -> List[UserPersona]:   
        user_personas = []
        for i in range(self.num_personas):
            persona_json = json_response[f"user_persona_{i}"]
            
            # Convert JSON to NumericAttribute and TextAttribute objects
            numeric_attributes = [
                NumericAttribute(
                    name=attr["name"],
                    value=attr["value"],
                    description=attr.get("description"),
                    min_value=attr.get("min_value"),
                    max_value=attr.get("max_value")
                ) for attr in persona_json["numeric_attributes"]
            ]
            
            text_attributes = [
                TextAttribute(
                    name=attr["name"],
                    value=attr["value"],
                    description=attr.get("description")
                ) for attr in persona_json["text_attributes"]
            ]
            
            # Create the UserPersona object
            user_persona = UserPersona(
                name=persona_json["name"],
                overview=persona_json["overview"],
                numeric_attributes=numeric_attributes,
                text_attributes=text_attributes
            )
            
            user_personas.append(user_persona)
        
        return user_personas
