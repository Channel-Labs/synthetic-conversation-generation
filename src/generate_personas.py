import argparse
import logging
from typing import List
from anthropic import Anthropic
from openai import OpenAI

from data_models.assistant import Assistant
from data_models.character_card import CharacterCard
from data_models.data_generation_config import DataGenerationConfig, ConversationLength

from llm_queries.llm_query import ModelProvider, OpenAIModelProvider, AnthropicModelProvider, BedrockModelProvider
from llm_queries.user_persona_query import UserPersonaQuery

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
    
    def __init__(self, model_provider: ModelProvider, model_id: str, assistant: Assistant, previous_personas: List[CharacterCard]):
        self.model_provider = model_provider
        self.model_id = model_id
        self.assistant = assistant
        self.previous_personas = previous_personas

    def generate_persona(self) -> CharacterCard:
        user_persona_generator = UserPersonaQuery(self.model_provider, self.model_id, self.assistant, self.previous_personas)
        return user_persona_generator.query()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant-name", type=str, required=True)
    parser.add_argument("--assistant-description", type=str, required=True)
    parser.add_argument("--num-personas", type=int, default=5)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-provider", type=str, choices=["openai", "anthropic", "bedrock"], default="openai")
    parser.add_argument("--user-persona-model", type=str, default="o4-mini")
    args = parser.parse_args()

    if args.model_provider == "openai":
        openai_client = OpenAI()
        model_provider = OpenAIModelProvider(openai_client)

    assistant = Assistant(name=args.assistant_name, description=args.assistant_description)

    persona_generator = PersonaGenerator(model_provider, args.user_persona_model, assistant, list())
    for _ in range(args.num_personas):
        persona = persona_generator.generate_persona()
        persona_generator.previous_personas.append(persona)

    # TODO: Don't hardcode conversation length
    data_generation_config = DataGenerationConfig(assistant, persona_generator.previous_personas, ConversationLength(min_turns=1, max_turns=5))
    data_generation_config.to_yaml(args.output_path)