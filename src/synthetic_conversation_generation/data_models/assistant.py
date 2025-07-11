import json
import yaml
from dataclasses import dataclass
from typing import Dict


@dataclass
class Assistant:
    name: str
    description: str

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Create an Assistant instance from a dictionary.
        
        Args:
            data: Dictionary containing assistant data
            
        Returns:
            Assistant instance
        """
        return cls(
            name=data['name'],
            description=data['description']
        )

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Create an Assistant instance from a YAML file.
        
        Args:
            yaml_path: Path to the YAML file containing assistant data
            
        Returns:
            Assistant instance
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @property
    def prompt_object(self) -> dict:
        return {"name": self.name, "description": self.description}

    @property
    def prompt_format(self) -> str:
        return json.dumps(self.prompt_object, indent=4)