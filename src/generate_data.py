import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm

from anthropic import Anthropic
from openai import OpenAI

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
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-schema-path", type=str, required=True)
    parser.add_argument("--destination", type=str, choices=["amplitude", "posthog"], required=True)
    parser.add_argument("--model-provider", type=str, choices=["openai", "anthropic", "bedrock"], default="openai")
    parser.add_argument("--event-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--event-property-model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--explanation-model", type=str, default="gpt-4.1-mini")
    args = parser.parse_args()

    if args.model_provider == "openai":
        openai_client = OpenAI()
        model_provider = OpenAIModelProvider(openai_client)
    # elif args.model_provider == "anthropic":
    #     anthropic_client = Anthropic()
    #     model_provider = AnthropicModelProvider(anthropic_client)
    # elif args.model_provider == "bedrock":
    #     bedrock_client = boto3.client("bedrock-runtime")
    #     model_provider = BedrockModelProvider(bedrock_client)

    data_schema = DataSchema.from_yaml(args.data_schema_path)

    ## Generate data
    ### Synthetic Data Generation Script