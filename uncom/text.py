import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')

PROMPT = 'User will provide you a transcription JSON from Whisper. Extract from it the object (noun + optional adjectives), the action (verb or phrase), and the target (noun + optional adjectives). If the action is a phrasal verb, put it whole. Return the result as JSON with the keys "object", "action", and "target". If you can\'t find any of these, leave the value empty. Be as concise as possible. Example: {"object": {"text": "mug", "timestamp": [1.04, 1.36]}, "action": {"text": "put on top", "timestamp": [1.5, 1.76]}, "target": {"text": "laptop", "timestamp": [2.24, 2.46]}}. Choose only one interpretation and write just one valid JSON object.'
PROMPT2 = 'Refine your own output to include information whether the object and the target are concrete objects like "apple" or not concrete like "here". Add appropriate "concrete" flag to your generated JSON.'
# PROMPT3 = 'Refine your second output if action or target imples that the target is to the left, to the right, in front or behind an object, change the "text" flag of "target" to \'<object>|<relative position>\', where relative position can only be  left, right, above or under. Example \'put to the right of the orange\' becomes \'orange|right\'. Do not create new flags.'

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
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch_dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

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
        # output = output.replace("```json", "").replace("```", "").strip()

        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "system", "content": PROMPT2})
        # messages.append({"role": "system", "content": PROMPT3})

        output = self.pipe(
            messages, max_new_tokens=600, return_full_text=False, do_sample=False
        )[0]["generated_text"]

        print("Generated text:", output)

        # Remove "```json" and "```" if there. Sometimes the model adds it.
        output = output.replace("```json", "").replace("```", "").strip()

        # Parse the output
        command = Command.from_text(output)

        return command

def check_relative_position(text):
    for position in ["left", "right", "beside", "between", "front", "behind"]:
        if position in text:
            return position 
    return False
