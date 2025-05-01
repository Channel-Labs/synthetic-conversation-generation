import argparse
import logging
from typing import List, Optional
from anthropic import Anthropic
from openai import OpenAI

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation_characters import ConversationCharacters

from synthetic_conversation_generation.llm_queries.llm_query import ModelProvider, OpenAIModelProvider, AnthropicModelProvider
from synthetic_conversation_generation.llm_queries.user_persona_query import UserPersonaQuery

# Configure root logger to WARNING to silence third-party libraries
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set loggers within this application to INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.WARNING)

class PersonaGenerator:
    
    def __init__(self, model_provider: ModelProvider, model_id: str, assistant: Assistant, previous_personas: List[CharacterCard], persona_guidance: Optional[str]=None):
        self.model_provider = model_provider
        self.model_id = model_id
        self.assistant = assistant
        self.previous_personas = previous_personas
        self.persona_guidance = persona_guidance

    def generate_persona(self) -> CharacterCard:
        user_persona_generator = UserPersonaQuery(self.model_provider, self.model_id, self.assistant, self.previous_personas, self.persona_guidance)
        return user_persona_generator.query()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant-name", type=str, required=True)
    parser.add_argument("--assistant-description", type=str, required=True)
    parser.add_argument("--num-personas", type=int, default=5)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-provider", type=str, choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model-id", type=str, default="o4-mini")
    args = parser.parse_args()

    if args.model_provider == "openai":
        openai_client = OpenAI()
        model_provider = OpenAIModelProvider(openai_client)
    else:
        anthropic_client = Anthropic()
        model_provider = AnthropicModelProvider(anthropic_client)

    assistant = Assistant(name=args.assistant_name, description=args.assistant_description)

    persona_generator = PersonaGenerator(model_provider, args.model_id, assistant, list())
    for _ in range(args.num_personas):
        persona = persona_generator.generate_persona()
        persona_generator.previous_personas.append(persona)

    conversation_characters = ConversationCharacters(assistant, persona_generator.previous_personas)
    conversation_characters.to_yaml(args.output_path)