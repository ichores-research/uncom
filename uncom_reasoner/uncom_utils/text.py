import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import chardet

# PROMPT = 'User will provide you a transcription JSON from Whisper. Extract from it the object (noun + optional adjectives), the action (verb or phrase), and the target (noun + optional adjectives). If the action is a phrasal verb, put it whole. Return the result as JSON with the keys "object", "action", and "target". If you can\'t find any of these, leave the value empty. Be as concise as possible. Example: {"object": {"text": "mug", "timestamp": [1.04, 1.36]}, "action": {"text": "put on top", "timestamp": [1.5, 1.76]}, "target": {"text": "laptop", "timestamp": [2.24, 2.46]}}. Choose only one interpretation and write just one valid JSON object.'
# PROMPT = 'User will provide you a transcription JSON from Whisper. Extract from it the object (noun + optional adjectives), the action (verb or phrase), and the target (noun + optional adjectives). If the target has a description of "next to", "between", "near", etc; it should be included in the target description. If the action is a phrasal verb, put it whole. Return the result as JSON with the keys "object", "action", and "target". If you can\'t find any of these, leave the value empty. Be as concise as possible. Example: {"object": {"text": "mug", "timestamp": [1.04, 1.36]}, "action": {"text": "put on top", "timestamp": [1.5, 1.76]}, "target": {"text": "laptop", "timestamp": [2.24, 2.46]}}. Choose only one interpretation and write just one valid JSON object.'
# PROMPT = 'User will provide you a transcription JSON from Whisper. Extract from it the object (noun + adjectives), the action (verb or phrase), and the target (noun + adjective). If the target has a description of "next to", "between", "near", etc; it should be included in the target description. If the action is a phrasal verb, put it whole. Return the result as JSON with the keys "object", "action", and "target". If you can\'t find any of these, leave the value empty. Be as concise as possible. Example: {"object": {"text": "red mug", "timestamp": [1.04, 1.36]}, "action": {"text": "put on top", "timestamp": [1.5, 1.76]}, "target": {"text": "laptop", "timestamp": [2.24, 2.46]}}. Choose only one interpretation and write just one valid JSON object.'

PROMPT = """You will receive a Whisper transcription JSON. Extract exactly one clear instance of:
- an object (noun + optional adjectives) or a reference to one, like " this", "that", "it";
- an action (verb or phrasal verb),
- a target (noun + optional adjectives, including any positional descriptors like "next to", "between", "near", etc.), or a reference to a target, like "this", "here" or "there".

Return a single JSON object with the keys: "object", "action", and "target". Each key should contain a "text" string and a single "timestamp" [start, finish] from the original transcription. If any element is missing or unclear, leave its value empty.

Follow this example strictly:

{"object": {"text": "red mug", "timestamp": [1.04, 1.36]}, "action": {"text": "put on top", "timestamp": [1.5, 1.76]}, "target": {"text": "laptop", "timestamp": [2.24, 2.46]}}

Output only the valid formatted JSON structure. No extra text or formatting.
 """

PROMPT2 = """ Refine your own output to include information whether the object and the target are concrete, specific entities (e.g., "apple", "chair") or a non-concrete reference such as "this", "that", "it".

Add a field "concrete": true if the object/target is a specific, tangible noun, and "concrete": false if it is a vague or referential term like "this".

Ensure that pronouns without a clearly identifiable object are marked as "concrete": false.
"""
# PROMPT_QUANTITY = """Refine your own output, enriching the "object" list by adding a `"quantity"` field to each item.

# Instructions:
# - For each object in the list, determine how many of that object the user wants the robot to pick.
# - If the user refers to a specific number (e.g., "two apples", "three mugs"), set `"quantity"` to that number.
# - If the user refers to all of the objects (e.g., "all the cups", "every spoon"), set `"quantity"` to `null`.
# - Transform the text field of all objects to singular and remore any reference to quantity.
# - If the object is plural but quantity is not specified, assume it is `null`.
# - If the object is singular and no quantity is specified, assume it is `1`.

# Return the updated JSON object with the `"quantity"` field added to each object. Do not change the structure of the rest of the JSON. Do not include any explanation—just return the updated JSON.
# """

PROMPT3 = """Determine whether the input text contains a complete command that includes:

        1. A single object to be picked up (or a reference to an object, like "this" or "that");

        2. An action to be performed with that object;

        3. A destination where the action should occur (or a reference to the location, like "here", "there, "over there", etc).

    Note: The object and destination may be referred to using pronouns or deictic terms like "this", "that", or "here".

    Respond only with True or False. Do not include any explanation or additional text.
 """

@dataclass
class Word:
    """
    A word extracted from the transcription.
    
    Attributes:
        text: The text of the word.
        timestamp: The start and end timestamps of the word in seconds.
        concrete: Whether the word is a concrete object.
    """
    text: str
    timestamp: tuple([float, float])
    concrete: Optional[bool] = None


@dataclass
class Command:
    """
    A command extracted from the transcription.
    """
    object: Word
    target: Word
    action: Word

    @classmethod
    def from_text(cls, extractor_text: str) -> "Command":

        print("THIS IS THE EXTRACTOR TEXT: ", chardet.detect(extractor_text.encode()))
        raw_string = repr(extractor_text)
        print(f"\n\n\n\nRaw String: {raw_string}\n\n\n\n")

        extractor = json.loads(extractor_text)
        return cls(
            object=Word(**extractor["object"]),
            target=Word(**extractor["target"]),
            action=Word(**extractor["action"]),
        )

    def to_json_str(self) -> str:
        return json.dumps(
            {
                "object": {
                    "text": self.object.text,
                    "timestamp": self.object.timestamp,
                },
                "action": {
                    "text": self.action.text,
                    "timestamp": self.action.timestamp,
                },
                "target": {
                    "text": self.target.text,
                    "timestamp": self.target.timestamp,
                },
            }
        )

    def save(self, path: Union[str, Path]):
        with open(path, "w") as f:
            f.write(self.to_json_str())


class CommandExtractor:
    """
    Extracts a command from a transcription.
    """
    def __init__(self, device="cuda", torch_dtype="auto") -> None:
        # Phi-mini model
        model = AutoModelForCausalLM.from_pretrained( "microsoft/Phi-3-mini-4k-instruct",
            # "microsoft/Phi-4-mini-instruct",
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
    
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    def extract(self, transcription):
        # trigram = n_gram_generator(transcription["text"], n=3, remove_stopwords=True)
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": str(transcription)},
        ]
        output = self.pipe(
            messages, max_new_tokens=500, return_full_text=False, do_sample=False
        )[0]["generated_text"]

        print("Generated text:", output)

        # Remove "```json" and "```" if there. Sometimes the model adds it.
        output = output.replace("```json", "").replace("```", "").strip()
        output = output.replace("(", "[").replace(")", "]")
        output = output.replace("None", "null")
        
        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "system", "content": PROMPT2})
        # messages.append({"role": "system", "content": PROMPT3})

        output = self.pipe(
            messages, max_new_tokens=600, return_full_text=False, do_sample=False
        )[0]["generated_text"]

        print("Generated text:", output)

        # Remove "```json" and "```" if there. Sometimes the model adds it.
        output = output.replace("```json", "").replace("```", "").strip()
        output = output.replace("(", "[").replace(")", "]")
        output = output.replace("None", "null")
        
        print("\n\n\n", "THIS IS THE OUTPUT AFTER STRIPPING", output, "\n\n\n")

        # messages.append({"role": "assistant", "content": output})
        # messages.append({"role": "system", "content": PROMPT_QUANTITY})
        # # messages.append({"role": "system", "content": PROMPT3})

        # output = self.pipe(
            #     messages, max_new_tokens=600, return_full_text=False, do_sample=False
        # )[0]["generated_text"]

        # print("Generated text 2:", output)

        # # Remove "```json" and "```" if there. Sometimes the model adds it.
        # output = output.replace("```json", "").replace("```", "").strip()

        # Parse the output
        command = Command.from_text(output)

        return command

    def check_command_completeness(self, text):
        messages = [
            {"role": "system", "content": PROMPT3},
            {"role": "user", "content": str(text)},
        ]
        output = self.pipe(
            messages, max_new_tokens=500, return_full_text=False, do_sample=False
        )[0]["generated_text"]
        
        return 'True' in output or 'true' in output


def check_relative_position(text):
    for position in ["left", "right", "next", "beside", "between", "front", "behind", "near", "close" ]:
        if position in text:
            return position 
    return False


def check_agreement(transcription, device="cuda", torch_dtype="auto"):
    print(f"HEARD: {transcription}")
    model = AutoModelForCausalLM.from_pretrained( #"microsoft/Phi-3-mini-4k-instruct",
                                                 "microsoft/Phi-4-mini-instruct",
                                                 trust_remote_code=True,
                                                 device_map=device,
                                                 torch_dtype=torch_dtype,
                                                )

    #tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
    
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                   )

    messages = [
        {"role": "system", "content": "Does the following phrase contain agreement, consent or Authorization? Answer only with '1' if true, '0' otherwise, nothing else."},
        {"role": "user", "content": str(transcription)},
    ]
    output = pipe(
        messages, max_new_tokens=100, return_full_text=False, do_sample=False
    )[0]["generated_text"].replace("'","")

    return output
