import argparse
import json
import logging
import random
from typing import List
from anthropic import Anthropic
from openai import OpenAI

from data_models.assistant import Assistant
from data_models.character_card import CharacterCard
from data_models.conversation import Conversation
from data_models.conversation_characters import ConversationCharacters
from data_models.inference_endpoint import InferenceEndpoint
from llm_queries.llm_query import LLMQuery, ModelProvider, OpenAIModelProvider, AnthropicModelProvider
from llm_queries.user_message_query import UserMessageQuery

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

class ConversationGenerator:

    def __init__(self, model_provider: ModelProvider, model_id: str, assistant_endpoint: InferenceEndpoint, assistant: Assistant, user_persona: CharacterCard):
        self.model_provider = model_provider
        self.model_id = model_id
        self.assistant_endpoint = assistant_endpoint
        self.assistant = assistant
        self.user_persona = user_persona

    def generate_conversation(self, conversation_length: int, conversation_id: str) -> Conversation:
        conversation = Conversation(
            id=str(conversation_id),
            user_id=self.user_persona.name,
            messages=[]
        )

        for _ in range(conversation_length):
            # Generate user message
            user_message_generator = UserMessageQuery(
                model_provider=self.model_provider,
                model_id=self.model_id,
                conversation=conversation,
                user_persona=self.user_persona,
                assistant=self.assistant
            )
            user_message = user_message_generator.query()
            conversation.messages.append(user_message)

            # Generate assistant response using inference endpoint
            assistant_message = self.assistant_endpoint.get_assistant_message(conversation)
            conversation.messages.append(assistant_message)

        return conversation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation-characters-path", type=str, required=True)
    parser.add_argument("--inference-endpoint-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-provider", type=str, choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model-id", type=str, default="gpt-4.1")
    parser.add_argument("--min-conversation-length", type=int, default=1)
    parser.add_argument("--max-conversation-length", type=int, default=5)
    args = parser.parse_args()

    if args.model_provider == "openai":
        openai_client = OpenAI()
        model_provider = OpenAIModelProvider(openai_client)
    # elif args.model_provider == "anthropic":
    #     anthropic_client = Anthropic()
    #     model_provider = AnthropicModelProvider(anthropic_client)

    convo_characters = ConversationCharacters.from_yaml(args.conversation_characters_path)
    assistant = convo_characters.assistant
    user_personas = convo_characters.users

    inference_endpoint = InferenceEndpoint.from_yaml(args.inference_endpoint_path)

    ## Generate synthetic data for each user persona
    conversations = []
    for conversation_id, user_persona in enumerate(user_personas):
        logger.info(f"Generating conversation {conversation_id} for user {user_persona.name}")

        # Randomly determine conversation length for this persona
        conversation_length = random.randint(args.min_conversation_length, args.max_conversation_length)

        conversation_generator = ConversationGenerator(model_provider, args.model_id, inference_endpoint, assistant, user_persona)
        conversation = conversation_generator.generate_conversation(conversation_length, conversation_id)
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


