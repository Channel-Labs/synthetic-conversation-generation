import argparse
import json
import os 
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

import openai
from openai import OpenAI

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.conversation import Conversation, Message, ROLE
from synthetic_conversation_generation.llm_queries.llm_query import OpenAIModelProvider
from synthetic_conversation_generation.llm_queries.create_grading_rubric_query import CreateGradingRubricQuery
from synthetic_conversation_generation.llm_queries.judge_conversation_query import JudgeConversationQuery


def format_conversation(prompt_messages, response_messages) -> Conversation:    
    messages = []
    for i, msg in enumerate(prompt_messages):
        messages.append(Message(role=ROLE.user, content=msg, timestamp=0, message_id=2*i))
        messages.append(Message(role=ROLE.assistant, content=response_messages[i], timestamp=0, message_id=2*i+1))
    
    return Conversation(id=0, user_id=0, messages=messages)


def calculate_score(prompt_messages, response_messages):
  
  conversation = format_conversation(prompt_messages, response_messages)

  return JudgeConversationQuery(
    model_provider=model_provider,
    model_id=args.model_id,
    grading_rubric=grading_rubric,
    assistant=assistant,
    conversation=conversation
  ).query()

def determine_winner(prompt_messages, response_a_messages, response_b_messages, num_shuffles=2):

  num_a_wins = 0
  num_b_wins = 0
  num_ties = 0

  for _ in range(num_shuffles):
    score_a = calculate_score(prompt_messages, response_a_messages)
    score_b = calculate_score(prompt_messages, response_b_messages)

    if score_a > score_b:
      num_a_wins += 1
    elif score_b > score_a:
      num_b_wins += 1
    else:
      num_ties += 1

  if num_a_wins == num_shuffles:
    return "a"
  elif num_b_wins == num_shuffles:
    return "b"
  else:
    return "tie"

  
def process_row(row):
    prompt_messages = json.loads(row['prompt'])
    response_a_messages = json.loads(row['response_a'])
    response_b_messages = json.loads(row['response_b'])
    
    return determine_winner(prompt_messages, response_a_messages, response_b_messages)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--assistant-name", type=str, required=True)
  parser.add_argument("--assistant-description", type=str, required=True)
  parser.add_argument("--chatbot-arena-conversations-file", type=str, required=True)
  parser.add_argument("--output-file", type=str, required=True)
  parser.add_argument("--num-conversations", type=int, default=200)
  parser.add_argument("--model-id", type=str, default="o3")
  args = parser.parse_args()

  assistant = Assistant(name=args.assistant_name, description=args.assistant_description)
  df = pd.read_csv(args.chatbot_arena_conversations_file)[:args.num_conversations]

  openai_client = OpenAI()
  model_provider = OpenAIModelProvider(openai_client)

  grading_rubric = CreateGradingRubricQuery(
    model_provider=model_provider,
    model_id=args.model_id,
    assistant=assistant
  ).query()
  print(f"Grading rubric: {grading_rubric}")

  predicted_winners = list()
  # Use ThreadPoolExecutor with pool size of 4
  with ThreadPoolExecutor(max_workers=4) as executor:
      # Submit all tasks
      future_to_index = {executor.submit(process_row, row): i for i, row in df.iterrows()}
      
      # Initialize results list with None values to maintain order
      results = [None] * len(df)
      
      # Collect results as they complete
      for future in tqdm.tqdm(as_completed(future_to_index), total=len(df)):
          index = future_to_index[future]
          try:
              winner = future.result()
              results[index] = winner
          except Exception as exc:
              print(f'Row {index} generated an exception: {exc}')
              results[index] = None  # or handle error as appropriate
      
      # Replace None results with "error" to maintain equal lengths of predicted and actual winners
      predicted_winners = [result if result is not None else "error" for result in results]

  actual_winners = list()
  for i, row in df.iterrows():
    if row['winner_model_a'] == 1:
      actual_winners.append("a")
    elif row['winner_model_b'] == 1:
      actual_winners.append("b")
    else:
      actual_winners.append("tie")

  print(f"Predicted winners length: {len(predicted_winners)}, Actual winners length: {len(actual_winners)}")
  print("Num correct: ", sum([predicted_winners[i] == actual_winners[i] for i in range(len(predicted_winners))]))

  df['predicted_winner'] = predicted_winners
  df['actual_winner'] = actual_winners
  df.to_csv(args.output_file, index=False)