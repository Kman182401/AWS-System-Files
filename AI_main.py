import json
import os
import openai
from dotenv import load_dotenv
from src.prompt_id_config import PROMPT_ID, PROMPT_VERSION
from src.cache import get_cached_response, set_cached_response
from src.validator import validate_code
from src.logging_config import logger

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

def determine_target_files(instruction):
    """
    Determines which files should be edited based on the user's instruction.
    Returns a list of file paths.
    """
    system_keywords = [
        "assistant system", "assistant", "main.py", "helper", "upgrade assistant",
        "conversational interface", "logging", "diff", "undo", "backup", "system feature"
    ]
    if any(kw in instruction.lower() for kw in system_keywords):
        # Only edit assistant files
        assistant_dir = "/home/ubuntu/codex_assistant/src/"
        return [os.path.join(assistant_dir, "main.py")]  # Add more as needed
    else:
        # Default: ask user for target file (pipeline)
        target_file = input("Enter path to the file you want to edit: ")
        return [target_file]

def present_and_confirm_changes(file_changes):
    """
    file_changes: dict of {filepath: (diff, explanation)}
    """
    print("\nProposed changes:")
    for filepath, (diff, explanation) in file_changes.items():
        print(f"\nFile to be changed: {filepath}")
        print("Explanation:")
        print(explanation)
        print("Diff preview:")
        print(diff)
    confirm = input("\nDo you want to apply these changes? (yes/no): ")
    return confirm.strip().lower() == "yes"

def main():
    print('Hi, let me know what I can assist you with. (Press Enter twice to finish your message.)')
    lines = []
    while True:
      line = input()
      if line == "":
          break
      lines.append(line)
    instruction = "\n".join(lines)
    cache_key = f"{PROMPT_ID}|{instruction}"
    cached = get_cached_response(cache_key)
    if cached and cached.strip():
        try:
            cached = json.loads(cached)
        except json.JSONDecodeError:
            print("Warning: Cached data is not valid JSON. Ignoring cache.")
            cached = None

    if cached:
        print("Using cached response.\n")
        ai_suggestion = cached
    else:
        client = openai.OpenAI()
        print("Querying OpenAI API...\n")
        response = client.responses.create(
            prompt={
                "id": PROMPT_ID,
                "version": PROMPT_VERSION
            },
            input=instruction
        )
        ai_suggestion = response.output
        # Try to get a 'content' attribute (for OpenAI message objects), otherwise fall back to str()
        suggestion_text = getattr(ai_suggestion, "content", str(ai_suggestion))
        set_cached_response(cache_key, suggestion_text)
    print("\n--- AI SUGGESTION ---\n")
    print(ai_suggestion)

    # Determine which files to edit based on the instruction
    target_files = determine_target_files(instruction)
    # For demonstration, we'll just show the AI suggestion as the "diff" and explanation
    file_changes = {}
    for file in target_files:
        diff = f"\n# --- AI Added code ---\n{ai_suggestion}"
        explanation = "This code will be added to the file as suggested by the AI."
        file_changes[file] = (diff, explanation)

    if present_and_confirm_changes(file_changes):
        for file, (diff, _) in file_changes.items():
            # Make a timestamped backup before editing
            import shutil, datetime
            backup_file = f"{file}.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            shutil.copy2(file, backup_file)
            print(f"Backup created: {backup_file}")

            with open(file, "a") as f:
                f.write(diff)
            print(f"AI suggestion appended to {file}.")

            print("\nRunning validation...")
            # Ensure ai_suggestion is a string before validation
            if isinstance(ai_suggestion, list):
                code_to_validate = "\n".join(str(line) for line in ai_suggestion)
            else:
                code_to_validate = str(ai_suggestion)
            is_valid, errors = validate_code(code_to_validate)
            if is_valid:
                print("Code validated successfully.")
            else:
                print(f"Validation errors:\n{errors}")
    else:
        print("No changes made.\n")

if __name__ == "__main__":
    main()

# --- AI Added code ---
[ResponseOutputMessage(id='msg_68946909dd74819eac24f0a18e0b4c5002b074d97b60ead2', content=[ResponseOutputText(annotations=[], text='### Reasoning and Step-by-Step Plan\n\nYou’ve requested a comprehensive upgrade to the assistant system, focusing on conversational flexibility, safe/controlled file editing, robust backup and diffing, code validation, logging, and minimal system impact. Here’s the step-by-step logic and plan:\n\n#### 1. Conversational Chat Int>



