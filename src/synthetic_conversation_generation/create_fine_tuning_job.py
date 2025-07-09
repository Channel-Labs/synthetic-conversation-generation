import os
import requests
import json

from synthetic_conversation_generation.llm_queries.create_grading_rubric_query import CreateGradingRubricQuery

def create_fine_tuning_job():
    """
    Create a fine-tuning job using OpenAI's API.
    
    Returns:
        dict: Response from the API
    """
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # API endpoint
    url = "https://api.openai.com/v1/fine_tuning/jobs"
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    judge_grader = {
      "type": "score_model",
      "name": "judge_grader",
      "input": [
            {
                "role": "user",
                "content": """Assess the assistant's performance in the following conversation on a scale of 0 to 100, based on the grading rubric provided.

### Grading Rubric
{{ sample.output_json.grading_rubric }}

### Assistant
{
    "name": "{{ item.assistant_name }}",
    "description": "{{ item.assistant_description }}"
}

### Conversation
{{ item.conversation_str }}        
"""
          }
      ],
      "model": "o3-2025-04-16",
      "range": [0, 100],
      "sampling_params": {
          "reasoning_effort": "medium"
      },
  }

    passthrough_grader = {
        "type": "python",
        "name": "passthrough_grader_v2",
        "source": "def grade(sample, item):\n    return int(item['expected_judge_score'])",
        "image_tag": "2025-05-08"
    }

    multi_grader = {
        "type": "multi",
        "graders": {
            "actual_score": judge_grader,
            "expected_score": passthrough_grader
        },
        "calculate_output": "(100 - abs(expected_score - actual_score)) / 100"
    }
        
    response_schema = CreateGradingRubricQuery(
        model_provider=None,
        model_id=None,
        assistant=None
    ).response_schema()

    training_file = "file-TobS1nmR7xCRtrmbaLSniU"
    validation_file = "file-LcFQV1AW2FbrZ76QGrzG2z"
    
    # Request payload
    data = {
        "training_file": training_file,
        "validation_file": validation_file,
        "model": "o4-mini-2025-04-16",
        "seed": 42,
        "method": {
            "type": "reinforcement",
            "reinforcement": {
                "grader": multi_grader,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": response_schema
                    }
                },
                "hyperparameters": {
                    "reasoning_effort": "medium",
                    "batch_size": 6,
                    "eval_interval": 4,
                    "n_epochs": 2
                }
            }
        }
    }
    
    try:
        # Make the POST request
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error creating fine-tuning job: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                print(f"Error details: {json.dumps(error_details, indent=2)}")
            except:
                print(f"Response text: {e.response.text}")
        raise


if __name__ == "__main__":
    try:
        result = create_fine_tuning_job()
        print("Fine-tuning job created successfully!")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Failed to create fine-tuning job: {e}")