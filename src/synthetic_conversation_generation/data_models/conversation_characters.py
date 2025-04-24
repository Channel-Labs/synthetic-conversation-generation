from dataclasses import dataclass, asdict
from typing import List

import yaml

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.character_card import CharacterCard


@dataclass
class ConversationCharacters:
    assistant: Assistant
    users: List[CharacterCard]

    @classmethod
    def from_yaml(cls, schema_path: str):
        """
        Load a YAML schema file and convert it into a ConversationCharacters object.
        
        Args:
            schema_path: Path to the YAML schema file
            
        Returns:
            ConversationCharacters object containing the parsed schema
        """
        with open(schema_path, 'r') as f:
            schema_data = yaml.safe_load(f)
        
        # Parse assistant
        assistant = Assistant.from_dict(schema_data['assistant'])
        
        # Parse users
        users = []
        for user_data in schema_data.get('users', []):
            users.append(CharacterCard.from_dict(user_data))
        
        return cls(
            assistant=assistant,
            users=users
        )
    
    def to_yaml(self, output_path: str):
        """
        Convert this ConversationCharacters object to YAML format and write it to the specified path.
        
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
    
