from dataclasses import dataclass
import json
from typing import List

import yaml

from data_models.assistant import Assistant
from data_models.user_persona import UserPersona, NumericAttribute, TextAttribute

@dataclass
class ConversationLength:
    min_turns: int
    max_turns: int

@dataclass
class DataGenerationConfig:
    assistant: Assistant
    user_personas: List[UserPersona]
    conversation_length: ConversationLength
    
    @classmethod
    def from_yaml(cls, schema_path: str):
        """
        Load a YAML schema file and convert it into a DataSchema object.
        
        Args:
            schema_path: Path to the YAML schema file
            
        Returns:
            DataSchema object containing the parsed schema
        """
        with open(schema_path, 'r') as f:
            schema_data = yaml.safe_load(f)
        
        # Parse assistant data
        assistant_data = schema_data['assistant']
        assistant = Assistant(
            name=assistant_data['name'],
            description=assistant_data['description']
        )
        
        # Parse user personas
        user_personas = []
        for persona_data in schema_data.get('user_personas', []):
            numeric_attributes = []
            for attr in persona_data.get('numeric_attributes', []):
                numeric_attributes.append(
                    NumericAttribute(
                        name=attr['name'],
                        description=attr.get('description'),
                        min_value=attr.get('min_value'),
                        max_value=attr.get('max_value'),
                        value=attr['value']
                    )
                )
            
            text_attributes = []
            for attr in persona_data.get('text_attributes', []):
                text_attributes.append(
                    TextAttribute(
                        name=attr['name'],
                        description=attr.get('description'),
                        value=attr['value']
                    )
                )
            
            user_personas.append(
                UserPersona(
                    name=persona_data['name'],
                    overview=persona_data['overview'],
                    numeric_attributes=numeric_attributes,
                    text_attributes=text_attributes
                )
            )

        # Parse conversation length
        conversation_length = ConversationLength(
            min_turns=schema_data['conversation_length']['min_turns'],
            max_turns=schema_data['conversation_length']['max_turns']
        )
        
        return cls(
            assistant=assistant,
            user_personas=user_personas,
            conversation_length=conversation_length
        )
    
