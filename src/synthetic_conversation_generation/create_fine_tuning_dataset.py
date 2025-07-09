import argparse
import random

import yaml
import json
from datetime import datetime
from uuid import uuid4
from tqdm import tqdm

from openai import OpenAI
import pandas as pd


from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation import Conversation, Message, ROLE
from synthetic_conversation_generation.llm_queries.llm_query import OpenAIModelProvider
from synthetic_conversation_generation.llm_queries.ground_truth_judge_query import GroundTruthJudgeQuery
from synthetic_conversation_generation.llm_queries.create_grading_rubric_query import CreateGradingRubricQuery
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_assistant_personas(yaml_file_path):
    """Load assistant personas from YAML file and return assistant and user personas."""
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    # Create Assistant object
    assistant_data = data['assistant']
    assistant = Assistant(
        name=assistant_data['name'],
        description=assistant_data['description']
    )
    
    # Create CharacterCard objects for each user persona
    user_personas = []
    for user_data in data['users']:
        user_persona = CharacterCard.from_dict(user_data)
        user_personas.append(user_persona)
    
    return assistant, user_personas


def load_conversations(jsonl_file_path):
    """Load conversations from JSONL file and return Conversation objects."""
    conversations = []
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            try:
                data = json.loads(line.strip())
                
                # Create Message objects
                messages = []
                for i, msg_data in enumerate(data['messages']):
                    role = ROLE.user if msg_data['role'] == 'user' else ROLE.assistant
                    message = Message(
                        role=role,
                        content=msg_data['content'],
                        timestamp=datetime.now(),
                        message_id=int(i)
                    )
                    messages.append(message)
                
                # Create Conversation object
                conversation = Conversation(
                    id=str(uuid4()),
                    user_id=f"user_{line_num}",
                    messages=messages
                )
                conversations.append(conversation)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return conversations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant-personas-file", type=str, required=True)
    parser.add_argument("--conversations-file", type=str, required=True)
    parser.add_argument("--num-attempts-per-conversation", type=int, default=5)
    parser.add_argument("--ground-truth-judge-model-id", type=str, default="o3")
    parser.add_argument("--rubric-generator-model-id", type=str, default="o4-mini")
    args = parser.parse_args()
    
    # Set up OpenAI client and model provider
    openai_client = OpenAI()
    model_provider = OpenAIModelProvider(openai_client)
    
    # Load assistant personas and conversations
    assistant, user_personas = load_assistant_personas(args.assistant_personas_file)
    conversations = load_conversations(args.conversations_file)

    grading_rubric_template_prompt = CreateGradingRubricQuery(
        model_provider=None,
        model_id=None,
        assistant=assistant,
    ).generate_prompt()

    results = []
    
    print(f"Loaded {len(user_personas)} user personas and {len(conversations)} conversations")

    dataset = []
    for idx, (user_persona, conversation) in enumerate(tqdm(zip(user_personas, conversations), total=len(user_personas), desc="Scoring pairs")):
        tqdm.write(f"Scoring: {user_persona.name} with conversation {conversation.user_id}")
        
        def get_score():
            # Re-generating the grading rubric for each conversation to increase diversity of selected rubrics
            grading_rubric = CreateGradingRubricQuery(
                model_provider=model_provider,
                model_id=args.rubric_generator_model_id,
                assistant=assistant
            ).query()

            score = GroundTruthJudgeQuery(
                model_provider=model_provider,
                model_id=args.ground_truth_judge_model_id,
                grading_rubric=grading_rubric,
                assistant=assistant,
                conversation=conversation,
                user_persona=user_persona
            ).query()
            
            return score, grading_rubric
        
        with ThreadPoolExecutor(max_workers=args.num_attempts_per_conversation) as executor:
            futures = [executor.submit(get_score) for _ in range(args.num_attempts_per_conversation)]
            results = [future.result() for future in as_completed(futures)]
        
        scores = [result[0] for result in results]
        grading_rubrics = [result[1] for result in results]
        
        avg_score = int(round(sum(scores) / len(scores)))
        tqdm.write(f"Scores: {scores} -> Averaged: {avg_score}")

        dataset.append({
            "messages": [
                {"role": "user", "content": grading_rubric_template_prompt}
            ],
            "assistant_name": assistant.name,
            "assistant_description": assistant.description,
            "conversation_str": json.dumps(conversation.prompt_format, indent=4),
            "expected_judge_score": avg_score,
            "grading_rubric": grading_rubrics[0]  # Use a grading rubric from one of the attempts as a reference, won't actually be used for training
        })

    # Shuffle dataset with fixed seed for reproducibility
    random.seed(42)
    shuffled_dataset = dataset.copy()
    random.shuffle(shuffled_dataset)
    
    # Split dataset into training (75%) and validation (25%)
    split_index = int(len(shuffled_dataset) * 0.75)
    training_dataset = shuffled_dataset[:split_index]
    validation_dataset = shuffled_dataset[split_index:]
    
    print(f"Dataset split: {len(training_dataset)} training samples, {len(validation_dataset)} validation samples")
    
    # Write training dataset
    with open('./training_dataset.jsonl', 'w') as f:
        for item in training_dataset:
            f.write(json.dumps(item) + '\n')
    
    # Write validation dataset
    with open('./validation_dataset.jsonl', 'w') as f:
        for item in validation_dataset:
            f.write(json.dumps(item) + '\n')
            

