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
    
    def to_yaml(self, output_path: str):
        """
        Convert this DataGenerationConfig object to YAML format and write it to the specified path.
        
        Args:
            output_path: Path where the YAML file will be written
            
        Returns:
            str: YAML representation of the config
        """
        data = {
            'assistant': {
                'name': self.assistant.name,
                'description': self.assistant.description
            },
            'user_personas': [],
            'conversation_length': {
                'min_turns': self.conversation_length.min_turns,
                'max_turns': self.conversation_length.max_turns
            }
        }
        
        # Convert user personas to dictionaries
        for persona in self.user_personas:
            persona_dict = {
                'name': persona.name,
                'overview': persona.overview,
                'numeric_attributes': [],
                'text_attributes': []
            }
            
            # Convert numeric attributes
            for attr in persona.numeric_attributes:
                attr_dict = {'name': attr.name, 'value': attr.value}
                if attr.description:
                    attr_dict['description'] = attr.description
                if attr.min_value is not None:
                    attr_dict['min_value'] = attr.min_value
                if attr.max_value is not None:
                    attr_dict['max_value'] = attr.max_value
                persona_dict['numeric_attributes'].append(attr_dict)
            
            # Convert text attributes
            for attr in persona.text_attributes:
                attr_dict = {'name': attr.name, 'value': attr.value}
                if attr.description:
                    attr_dict['description'] = attr.description
                persona_dict['text_attributes'].append(attr_dict)
            
            data['user_personas'].append(persona_dict)
        
        yaml_content = yaml.dump(data, sort_keys=False)
        
        # Write to the output file
        with open(output_path, 'w') as f:
            f.write(yaml_content)
            
        return yaml_content
    
