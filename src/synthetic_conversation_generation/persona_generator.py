import argparse
import logging
from typing import List, Optional
import yaml

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
    
    def __init__(self, model_provider: ModelProvider, model_id: str, assistant: Assistant, previous_personas: List[CharacterCard]):
        self.model_provider = model_provider
        self.model_id = model_id
        self.assistant = assistant
        self.previous_personas = previous_personas

    def generate_persona(self) -> CharacterCard:
        user_persona_generator = UserPersonaQuery(self.model_provider, self.model_id, self.assistant, self.previous_personas)
        return user_persona_generator.query(max_retries=1, timeout=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant-path", type=str, required=True, help="Path to YAML file containing assistant definition")
    parser.add_argument("--num-personas", type=int, default=5, help="Number of user personas to generate")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the generated personas (YAML format)")
    parser.add_argument("--model-provider", type=str, choices=["openai", "anthropic"], default="openai", help="LLM provider to use for generating personas")
    parser.add_argument("--model-id", type=str, default="o3", help="Model ID for persona generation")
    parser.add_argument("--previous-personas-path", type=str, help="Path to YAML file containing previous personas to avoid duplication")
    args = parser.parse_args()

    if args.model_provider == "openai":
        openai_client = OpenAI()
        model_provider = OpenAIModelProvider(openai_client)
    else:
        anthropic_client = Anthropic()
        model_provider = AnthropicModelProvider(anthropic_client)

    assistant = Assistant.from_yaml(args.assistant_path)

    # Load previous personas if provided
    previous_personas = []
    if args.previous_personas_path:
        logger.info(f"Loading previous personas from {args.previous_personas_path}")
        conversation_characters = ConversationCharacters.from_yaml(args.previous_personas_path)
        previous_personas = conversation_characters.users
        logger.info(f"Loaded {len(previous_personas)} previous personas")

    persona_generator = PersonaGenerator(model_provider, args.model_id, assistant, previous_personas)
    
    # Generate new personas
    new_personas = []
    for i in range(args.num_personas):
        print(f"Generating persona {i+1} of {args.num_personas}")
        persona = persona_generator.generate_persona()
        new_personas.append(persona)
        persona_generator.previous_personas.append(persona)

    # Save only the new personas to the output file
    conversation_characters = ConversationCharacters(users=new_personas)
    conversation_characters.to_yaml(args.output_path)