from typing import Dict, List, Optional

from dataclasses import dataclass


@dataclass
class CharacterCard:
    name: str
    description: str
    personality: str
    scenario: str


    # overview: str
    # numeric_attributes: List[NumericAttribute]
    # text_attributes: List[TextAttribute]

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Create a CharacterCard instance from a dictionary.
        
        Args:
            data: Dictionary containing character data
            
        Returns:
            CharacterCard instance
        """
        return cls(
            name=data['name'],
            description=data['description'],
            personality=data['personality'],
            scenario=data['scenario']
        )


    # {{input}}: User input insertion
# {{name}}: Character name
# {{description}}: Character description
# {{scenario}}: Scenario information
# {{first_mes}}: First message
# {{mes_example}}: Message examples
# {{personality}}: Personality traits


