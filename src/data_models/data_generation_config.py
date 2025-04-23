from dataclasses import dataclass, asdict
import json
from typing import List, Dict, Any

import yaml

from data_models.assistant import Assistant
from data_models.character_card import CharacterCard

@dataclass
class ConversationLength:
    min_turns: int
    max_turns: int
    
    @classmethod
    def from_dict(cls, data: Dict):
        """
        Create a ConversationLength instance from a dictionary.
        
        Args:
            data: Dictionary containing conversation length data
            
        Returns:
            ConversationLength instance
        """
        return cls(
            min_turns=data['min_turns'],
            max_turns=data['max_turns']
        )

@dataclass
class DataGenerationConfig:
    assistant: Assistant
    user_personas: List[CharacterCard]
    conversation_length: ConversationLength

    @classmethod
    def from_yaml(cls, schema_path: str):
        """
        Load a YAML schema file and convert it into a DataGenerationConfig object.
        
        Args:
            schema_path: Path to the YAML schema file
            
        Returns:
            DataGenerationConfig object containing the parsed schema
        """
        with open(schema_path, 'r') as f:
            schema_data = yaml.safe_load(f)
        
        # Parse assistant data
        assistant = Assistant.from_dict(schema_data['assistant'])
        
        # Parse user personas
        user_personas = []
        for persona_data in schema_data.get('user_personas', []):
            user_personas.append(CharacterCard.from_dict(persona_data))

        # Parse conversation length
        conversation_length = ConversationLength.from_dict(schema_data['conversation_length'])
        
        return cls(
            assistant=assistant,
            user_personas=user_personas,
            conversation_length=conversation_length
        )
    
    def to_yaml(self, output_path: str):
        """
        Convert this DataGenerationConfig object to YAML format and write it to the specified path.
        
        Args:
            output_path: Path where the YAML file will be written
            
        Returns:
            str: YAML representation of the config
        """
        # Convert the dataclass to a dictionary
        data = asdict(self)
        
        yaml_content = yaml.dump(data, sort_keys=False)
        
        # Write to the output file
        with open(output_path, 'w') as f:
            f.write(yaml_content)
            
        return yaml_content
    
