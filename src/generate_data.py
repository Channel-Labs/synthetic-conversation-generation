import argparse
import json
import logging
import random

from anthropic import Anthropic
from openai import OpenAI

from data_models.conversation import Conversation
from data_models.data_generation_config import DataGenerationConfig
from data_models.inference_endpoint import InferenceEndpoint
from llm_queries.llm_query import LLMQuery, ModelProvider, OpenAIModelProvider, AnthropicModelProvider, BedrockModelProvider
from llm_queries.user_message_generator import UserMessageGenerator

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
    parser.add_argument("--data-generation-config-path", type=str, required=True)
    parser.add_argument("--inference-endpoint-config-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-provider", type=str, choices=["openai", "anthropic", "bedrock"], default="openai")
    parser.add_argument("--user-message-model", type=str, default="gpt-4.1")
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

    data_generation_config = DataGenerationConfig.from_yaml(args.data_generation_config_path)
    assistant = data_generation_config.assistant
    user_personas = data_generation_config.user_personas
    conversation_length_params = data_generation_config.conversation_length

    inference_endpoint = InferenceEndpoint.from_yaml(args.inference_endpoint_config_path)

    ## Generate synthetic data for each user persona
    conversations = []
    for conversation_id, user_persona in enumerate(user_personas):
        # Randomly determine conversation length for this persona
        conversation_length = random.randint(conversation_length_params.min_turns, conversation_length_params.max_turns)
        
        conversation = Conversation(
            id=str(conversation_id),
            user_id=user_persona.name,
            messages=[]
        )
        
        for turn in range(conversation_length):
            # Generate user message
            user_message_generator = UserMessageGenerator(
                model_provider=model_provider,
                model_id=args.user_message_model,
                conversation=conversation,
                user_persona=user_persona,
                assistant=assistant
            )
            user_message = user_message_generator.query()
            conversation.messages.append(user_message)

            # Generate assistant response using inference endpoint
            assistant_message = inference_endpoint.get_assistant_message(conversation)
            conversation.messages.append(assistant_message)

        conversations.append(conversation)

    ## Save conversations to file
    with open(args.output_path, "w") as f:
        for conversation in conversations:
            # Convert each conversation to the desired format
            formatted_messages = []
            
            # Add user and assistant messages
            for message in conversation.messages:
                formatted_message = {
                    "role": message.role.name,
                    "content": message.content
                }
                formatted_messages.append(formatted_message)
            
            # Create the final jsonl object and write to file
            jsonl_object = {"messages": formatted_messages}
            f.write(json.dumps(jsonl_object) + "\n")


