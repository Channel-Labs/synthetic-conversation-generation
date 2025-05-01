from dataclasses import asdict
import json
from typing import List, Optional

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider

class UserPersonaQuery(LLMQuery):

    def __init__(
        self, 
        model_provider: ModelProvider, 
        model_id: str, 
        assistant: Assistant, 
        previous_personas: List[CharacterCard], 
        persona_guidance: Optional[str]=None
    ):
        super().__init__(model_provider, model_id)
        self.assistant = assistant
        self.previous_personas = previous_personas
        self.persona_guidance = persona_guidance

    def generate_prompt(self):
        return f"""Create a distinct, realistic, and well-defined user persona that represents someone likely to interact with the AI assistant defined below. You'll later use these personas to drive simulated conversations and evaluate the assistant's performance. Thus, each generated persona should fill a gap left by existing personas.

### Background
These personas will be utilized to generate simulated conversations and evaluate the performance of the AI assistant. Thus, the new persona should be distinct from the previous personas in order to test the assistant's performance across diverse user types, identify potential gaps in the assistant's response capabilities, and ensure comprehensive test coverage across a wide range of potential interactions.

### Instructions
1. Review the assistant definition, previous user personas, and persona guidance.
2. Invent a new persona (name, background, personality, goals, motivations, communication style, etc.) that is likely to seek out the defined assistant, as well as a scenario for why the user is seeking out this assistant's help.
3. Incorporate the provided persona guidance to shape your creation. If no guidance is provided, develop the persona based solely on filling gaps in the existing persona collection.

### Assistant Definition
{json.dumps(asdict(self.assistant), indent=4)}

### Previous User Personas
{json.dumps([asdict(persona) for persona in self.previous_personas], indent=4)}

### Persona Guidance
{self.persona_guidance if self.persona_guidance else ""}
"""
    
    def response_schema(self):
        properties = {
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
                "description": "The context and circumstances for why the user is interacting with the assistant."      
            },
            "summary": {
                "type": "string",
                "description": "A concise (~5 word) summary of the user, with the main focus on the user's scenario."
            }
        }

        return {
            "type": "object",
            "properties": properties,
            "required": ["name", "description", "personality", "scenario", "summary"],
            "additionalProperties": False
        }
    
    def parse_response(self, json_response) -> CharacterCard:   
        return CharacterCard(
            name=json_response["name"],
            description=json_response["description"],
            personality=json_response["personality"],
            scenario=json_response["scenario"],
            summary=json_response["summary"]
        )
