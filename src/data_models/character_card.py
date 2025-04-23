from typing import Dict, List, Optional

from dataclasses import dataclass


@dataclass
class CharacterCard:
    name: str
    description: str
    personality: str
    scenario: str

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


