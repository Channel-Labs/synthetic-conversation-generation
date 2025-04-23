import argparse
import json
import logging
import random

from anthropic import Anthropic
from openai import OpenAI

from data_models.assistant import Assistant
from data_models.data_generation_config import DataGenerationConfig, ConversationLength

from llm_queries.llm_query import LLMQuery, ModelProvider, OpenAIModelProvider, AnthropicModelProvider, BedrockModelProvider
from llm_queries.user_persona_generator import UserPersonaGenerator

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

    user_persona_generator = UserPersonaGenerator(model_provider, args.user_persona_model, assistant, args.num_personas)
    user_personas = user_persona_generator.query()

    # TODO: Don't hardcode conversation length
    data_generation_config = DataGenerationConfig(assistant, user_personas, ConversationLength(min_turns=1, max_turns=5))
    data_generation_config.to_yaml(args.output_path)