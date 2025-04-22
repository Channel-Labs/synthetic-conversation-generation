import json
from dataclasses import dataclass


@dataclass
class Assistant:
    name: str
    description: str

    @property
    def prompt_object(self) -> dict:
        return {"name": self.name, "description": self.description}

    @property
    def prompt_format(self) -> str:
        return json.dumps(self.prompt_object, indent=4)