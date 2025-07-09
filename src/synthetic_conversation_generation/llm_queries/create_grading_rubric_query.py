
from dataclasses import asdict
import json

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider

class CreateGradingRubricQuery(LLMQuery):

    def __init__(self, model_provider: ModelProvider, model_id: str, assistant: Assistant):
        super().__init__(model_provider, model_id)
        self.assistant = assistant
        
    def generate_prompt(self):
        prompt_template = f"""Assess the assistant's performance in the following conversation on a scale of 0 to 100, based on the grading rubric provided.

### Grading Rubric
{{GRADING_RUBRIC}}

### Assistant
{json.dumps(asdict(self.assistant), indent=4)}

### Conversation
{{CONVERSATION}}"""

        return f"""Generate the grading rubric that will enable the llm_judge_prompt_template to most accurately assess conversations.

<llm_judge_prompt_template>
{prompt_template}
</llm_judge_prompt_template>
"""

    def response_schema(self):
      properties = {
          "grading_rubric": {
              "type": "string",
              "description": "The grading rubric that should appear within the llm_judge_prompt_template"
          }
      }

      return {
          "type": "object",
          "properties": properties,
          "required": ["grading_rubric"],
          "additionalProperties": False
      }
    
    def parse_response(self, json_response) -> str:   
        return json_response["grading_rubric"]