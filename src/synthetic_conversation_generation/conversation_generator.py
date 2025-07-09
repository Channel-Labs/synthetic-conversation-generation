import argparse
import json
import logging
import random
from typing import List
from anthropic import Anthropic
from openai import OpenAI

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation import Conversation
from synthetic_conversation_generation.data_models.conversation_characters import ConversationCharacters
from synthetic_conversation_generation.data_models.inference_endpoint import InferenceEndpoint
from synthetic_conversation_generation.llm_queries.conversation_completion_query import ConversationCompletionQuery
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider, OpenAIModelProvider, AnthropicModelProvider
from synthetic_conversation_generation.llm_queries.user_message_query import UserMessageQuery

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

    def __init__(self, model_provider: ModelProvider, model_id: str, assistant_endpoint: InferenceEndpoint, assistant: Assistant, user_persona: CharacterCard, max_conversation_turns: int):
        self.model_provider = model_provider
        self.model_id = model_id
        self.assistant_endpoint = assistant_endpoint
        self.assistant = assistant
        self.user_persona = user_persona
        self.max_conversation_turns = max_conversation_turns

    def generate_conversation(self, conversation_id: str) -> Conversation:
        conversation = Conversation(
            id=str(conversation_id),
            user_id=self.user_persona.name,
            messages=[]
        )
        # Always start with a user message
        user_message_generator = UserMessageQuery(
            model_provider=self.model_provider,
            model_id=self.model_id,
            conversation=conversation,
            user_persona=self.user_persona,
            assistant=self.assistant
        )

        # Continue conversation until completion or max turns
        for i in range(self.max_conversation_turns - 1):
            print(f"Conversation turn: {i}")

            # Continue conversation with next user message
            user_message = user_message_generator.query()
            conversation.messages.append(user_message)

            # Generate assistant response
            assistant_message = self.assistant_endpoint.get_assistant_message(conversation)
            conversation.messages.append(assistant_message)

            # Check if conversation should end
            completion_checker = ConversationCompletionQuery(
                model_provider=self.model_provider,
                model_id="o3",
                conversation=conversation,
                user_persona=self.user_persona,
                assistant=self.assistant
            )
            is_complete = completion_checker.query()
            if is_complete:
                break

        return conversation    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversation-characters-path", type=str, required=True)
    parser.add_argument("--inference-endpoint-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-provider", type=str, choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model-id", type=str, default="gpt-4o")
    parser.add_argument("--max-conversation-turns", type=int, default=4)
    args = parser.parse_args()

    if args.model_provider == "openai":
        openai_client = OpenAI()
        model_provider = OpenAIModelProvider(openai_client)
    else:
        anthropic_client = Anthropic()
        model_provider = AnthropicModelProvider(anthropic_client)

    convo_characters = ConversationCharacters.from_yaml(args.conversation_characters_path)
    assistant = convo_characters.assistant
    user_personas = convo_characters.users

    inference_endpoint = InferenceEndpoint.from_yaml(args.inference_endpoint_path)

    ## Generate synthetic data for each user persona
    conversations = []
    for conversation_id, user_persona in enumerate(user_personas):
        logger.info(f"Generating conversation {conversation_id} for user {user_persona.name}")

        conversation_generator = ConversationGenerator(model_provider, args.model_id, inference_endpoint, assistant, user_persona, args.max_conversation_turns)
        conversation = conversation_generator.generate_conversation(conversation_id)
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


