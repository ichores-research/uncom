import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

PROMPT = 'User will provide you a transcription JSON from Whisper. Extract from it the object (noun + optional adjectives), the action (verb or phrase), and the target (noun + optional adjectives). Return the result as JSON with the keys "object", "action", and "target". If you can\'t find any of these, leave the value empty. Be as concise as possible. Additionally, determine whether object and target are concrete objects like "apple" or not concrete like "here". Add appriopriate "concrete" flag. Example: {"object": {"text": "mug", "concrete": "true", "timestamp": [1.04, 1.36]}, "action": {"text": "put on top", "timestamp": [1.5, 1.76]}, "target": {"text": "laptop", "concrete": "false", "timestamp": [2.24, 2.46]}}. Choose only one interpretation and write just one valid JSON object without additional text or special formatting.'


@dataclass
class Word:
    text: str
    timestamp: tuple[float, float]
    concrete: Optional[bool] = None


@dataclass
class Command:
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
    def __init__(self, device="cuda", torch_dtype="auto") -> None:
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
        messages = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": str(transcription)},
        ]
        output = self.pipe(
            messages, max_new_tokens=500, return_full_text=False, do_sample=False
        )[0]["generated_text"]

        print("Generated text:", output)

         # Remove "```json" and "```" if there
        output = output.replace("```json", "").replace("```", "").strip()

        command = Command.from_text(output)

        return command
