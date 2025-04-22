from typing import Dict, List, Optional

from dataclasses import dataclass


@dataclass
class NumericAttribute:
    name: str
    value: float
    description: Optional[str] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class TextAttribute:
    name: str
    value: str
    description: Optional[str] = None


# TODO: TURN THIS INTO STANDARD CHARACTER CARDS!!!
@dataclass
class UserPersona:
    name: str
    overview: str
    numeric_attributes: List[NumericAttribute]
    text_attributes: List[TextAttribute]


