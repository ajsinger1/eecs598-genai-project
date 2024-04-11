import pandas as pd
from llama_chat_helpers import format_prompt
prompts_with_output_path = '/content/drive/MyDrive/eecs598_genai/github_repo/data/prompts-with-output.json'
data = pd.read_json(prompts_with_output_path)
output_lengths = {format_prompt(data["prompt"][i]): float(data["output_length"][i]) for i in data.index}
print("\033[1;31;40m Using custom vLLM implementation with output length priority function")

def get_output_length(prompt: str) -> float:
    if prompt not in output_lengths:
        raise RuntimeError(f"Prompt {prompt} is not in output length map")
    return output_lengths[prompt] 