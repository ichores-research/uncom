import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Prompt sets — keyed by model family.
# Add a new key + dict to support additional models.
# ---------------------------------------------------------------------------

_PROMPTS_PHI3 = {
    "extract": """Robot pick-and-place parser. Extract ONE command from a Whisper transcription.
Return ONLY a JSON object:
{"object":{"text":"","descriptors":[],"location":null,"property_reference":null,"property_reference_target":null,"timestamp":[0.0,0.0],"concrete":true},"action":{"text":"","timestamp":[0.0,0.0]},"target":{"text":"","descriptors":[],"location":null,"property_reference":null,"property_reference_target":null,"timestamp":[0.0,0.0],"concrete":true}}
Rules: descriptors=visual only. location={"relation":"","reference":""}. property_reference=left/right/front/behind/next/near/between/color/shape/size. concrete=false for this/that/here/there. "" if absent.
""",

    "group_detect": """Robot pre-processor. Does the transcription refer to a GROUP of objects using plural/deictic words?
Return ONLY: {"is_group":bool,"descriptor":"singular noun or empty string","reference":"anchor object or null"}
is_group=true ONLY for explicit plurals or deictics: these/those/all the X/Xs/grab them all.
is_group=false for ANY singular command, even if a destination is mentioned.
reference=spatial anchor only (near/next to X), NOT the destination of a pick-and-place.
Examples:
"pick up these apples" -> {"is_group":true,"descriptor":"apple","reference":null}
"grab those" -> {"is_group":true,"descriptor":"","reference":null}
"bottles near the cup" -> {"is_group":true,"descriptor":"bottle","reference":"cup"}
"pick the red mug" -> {"is_group":false,"descriptor":null,"reference":null}
"pick up the mustard and put it in the bowl" -> {"is_group":false,"descriptor":null,"reference":null}
"put the apple in the box" -> {"is_group":false,"descriptor":null,"reference":null}
""",

    "extract_multi": """Robot pick-and-place parser. Extract ALL commands from a Whisper transcription.
Return ONLY a JSON array. One entry per physical action (pick, place, move).
For "pick up X and put it in Y" return ONE command: object=X, action=pick, target=Y.
Each command: {"object":{"text":"","descriptors":[],"location":null,"property_reference":null,"property_reference_target":null,"quantity":1,"timestamp":[0.0,0.0],"concrete":true},"action":{"text":"","timestamp":[0.0,0.0]},"target":{"text":"","descriptors":[],"location":null,"property_reference":null,"property_reference_target":null,"timestamp":[0.0,0.0],"concrete":true}}
Rules: object=thing being moved. target=destination. quantity=1(single)/N(count)/"these"(group deictic)/null(all).
Example: "pick up the mustard and put it in the bowl" -> [{"object":{"text":"mustard","descriptors":[],"location":null,"property_reference":null,"property_reference_target":null,"quantity":1,"timestamp":[0.0,3.0],"concrete":true},"action":{"text":"pick","timestamp":[0.0,1.0]},"target":{"text":"bowl","descriptors":[],"location":null,"property_reference":null,"property_reference_target":null,"timestamp":[3.0,5.0],"concrete":true}}]
""",

    "complete": """Is this robot voice command complete enough to execute? Reply ONLY: {"complete":bool,"missing":"object or target or action or none"}
""",

    "agreement": """Does this phrase express agreement or consent? Reply with exactly 1 or 0. Nothing else.
""",
}

_PROMPTS_QWEN = {
    "extract": """You are a language parser for a robot arm that performs pick-and-place tasks.

The user will give you a Whisper transcription JSON. Extract exactly ONE pick-and-place command and return a single valid JSON object. No markdown, no explanation, nothing else.

LOCATION objects always have exactly two string fields:
  {"relation": "<spatial preposition>", "reference": "<object name>"}

The full schema:

{
  "object": {
    "text":        <string>   Core noun or deictic reference ("mug", "this").
                              "" if absent.
    "descriptors": <array>    Visual/physical attributes only — color, size,
                              shape, material, pattern, state. [] if none.
    "location":    <object|null>
                              Where the object is BEFORE being moved.
                              {"relation": "next to", "reference": "keyboard"}
                              null if not stated.
    "property_reference":        <string|null>
                              HOW to find this entity when it cannot be identified
                              by name or descriptors alone.
                              One of: color/shape/size/left/right/front/behind/next/near/between
                              null if not applicable.
    "property_reference_target": <string|null>
                              The reference object(s) for property_reference.
                              For "between", join two nouns with " and ".
                              null if property_reference is null.
    "timestamp":   [start, end]   Floats from Whisper segments. [0.0,0.0] if unknown.
    "concrete":    <bool>     true  = specific tangible noun
                              false = deictic / pronoun ("this","it","that one")
  },

  "action": {
    "text":        <string>   Bare verb or phrasal verb ONLY.
                              "put on top of the box" -> "put"
                              "" if absent.
    "timestamp":   [start, end]
  },

  "target": {
    "text":        <string>   Core destination noun, singular. "" if absent.
    "descriptors": <array>    Same rules as object.descriptors. [] if none.
    "location":    <object|null>
                              WHERE the object should be placed relative to target.
                              null if none.
    "property_reference":        <string|null>   Same rules as object.property_reference.
    "property_reference_target": <string|null>   Same rules as object.property_reference_target.
    "timestamp":   [start, end]
    "concrete":    <bool>     true  = specific tangible destination
                              false = deictic ("here","there","over there")
  }
}

RULES:
1. descriptors = visual/physical ONLY. Never put spatial phrases in descriptors.
2. location = {"relation": <string>, "reference": <string>} or null.
3. If the action encodes placement: action.text="put", target.location={"relation":...}.
4. property_reference encodes HOW to find an object by spatial or visual property.
5. Timestamps: span of words from Whisper segments.
6. Deictic terms: concrete=false, copy verbatim into text.
7. Output ONLY the JSON object. Nothing before or after.
""",

    "group_detect": """You are a pre-processor for a robot pick-and-place system.

Detect whether a transcription refers to a GROUP of objects and extract:
- is_group: true if command refers to multiple objects as a group
- descriptor: the object type (singular, lowercased), "" if purely deictic
- reference: a named object anchoring the group spatially, null if gesture-only

Output ONLY a JSON object. Nothing before or after.

Rules:
1. is_group=true for: these/those/all the X/the Xs/these Xs/Xs near Y/grab them all.
2. is_group=false for singular: "the apple", "this mug", "it".
3. descriptor = singular lowercased noun. "" if purely deictic ("these", "those").
4. reference = named anchor object. null if located by gesture alone.

Examples:
Input:  "pick up these apples"
Output: {"is_group": true, "descriptor": "apple", "reference": null}

Input:  "grab those"
Output: {"is_group": true, "descriptor": "", "reference": null}

Input:  "get the bottles near the cup"
Output: {"is_group": true, "descriptor": "bottle", "reference": "cup"}

Input:  "pick up the red mug"
Output: {"is_group": false, "descriptor": null, "reference": null}

Input:  "move all the bananas next to the bowl"
Output: {"is_group": true, "descriptor": "banana", "reference": "bowl"}

Input:  "grab them all and place them there"
Output: {"is_group": true, "descriptor": "", "reference": null}

Input:  "pick up the apples and place them in the basket"
Output: {"is_group": true, "descriptor": "apple", "reference": null}
""",

    "extract_multi": """You are a language parser for a robot arm that performs pick-and-place tasks.

The user will give you a Whisper transcription JSON that may contain ONE or
MULTIPLE sequential commands. Return a JSON array of command objects — one
per distinct action. No markdown, no explanation, nothing else.

Each command uses this schema:
{
  "object": {
    "text": "", "descriptors": [], "location": null,
    "property_reference": null, "property_reference_target": null,
    "quantity": null, "timestamp": [0.0, 0.0], "concrete": true
  },
  "action": {"text": "", "timestamp": [0.0, 0.0]},
  "target": {
    "text": "", "descriptors": [], "location": null,
    "property_reference": null, "property_reference_target": null,
    "timestamp": [0.0, 0.0], "concrete": true
  }
}

quantity rules:
  null    = detect all instances of the object
  N       = integer count explicitly stated
  "these" = group deictic (these/those/them)
  1       = single object (default)

No object = place currently held object.
No target  = pick and hold, no place.
Output ONLY the JSON array.
""",

    "complete": """You are checking whether a voice command to a robot arm is complete.
A complete command needs at minimum an action and either an object or a target.
Reply ONLY with a JSON object: {"complete": bool, "missing": "object or target or action or none"}
Nothing else.
""",

    "agreement": """Does the following phrase express agreement, consent, or authorisation?
Respond with exactly one character: 1 if yes, 0 if no. Nothing else.
""",
}

# Map model family key -> prompt dict.
# _resolve_model_family() maps a model ID string to one of these keys.
PROMPTS = {
    "phi3":   _PROMPTS_PHI3,
    "qwen":   _PROMPTS_QWEN,
}

# Fallback for unknown models
_DEFAULT_PROMPT_FAMILY = "phi3"


def _resolve_model_family(model_id: str) -> str:
    """Map a HuggingFace model ID to a prompt family key."""
    mid = model_id.lower()
    if "qwen" in mid:
        return "qwen"
    if "phi" in mid:
        return "phi3"
    return _DEFAULT_PROMPT_FAMILY



# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpatialLocation:
    """
    A structured spatial relation between an entity and a reference object.

    Attributes:
        relation:  The spatial keyword/phrase, e.g. "next to", "on top of",
                   "between", "to the left of".
        reference: The noun phrase of the reference object, including any
                   descriptors, e.g. "keyboard", "tall green bottle".
    """
    relation: str
    reference: str

    @classmethod
    def from_dict(cls, d: dict) -> "SpatialLocation":
        return cls(
            relation=d.get("relation", ""),
            reference=d.get("reference", ""),
        )

    def to_dict(self) -> dict:
        return {"relation": self.relation, "reference": self.reference}

    def to_query(self) -> str:
        """Natural language form for passing to a detector or logger."""
        return f"{self.relation} {self.reference}".strip()


@dataclass
class EntityWord:
    """
    An object or target entity extracted from the transcription.

    Attributes:
        text:                     Core noun or deictic reference.
        descriptors:              Visual/physical attributes only.
        location:                 Structured spatial relation to a reference object.
        property_reference:       HOW to find this entity when name/descriptors
                                  are insufficient. One of:
                                    spatial: "left","right","front","behind",
                                             "next","near","between"
                                    visual:  "color","shape","size"
                                  None if not applicable.
        property_reference_target: The reference noun phrase for property_reference.
                                  For "between": two nouns joined with " and ".
                                  None if property_reference is None.
        timestamp:                (start, end) seconds from Whisper.
        concrete:                 True if specific tangible noun; False if deictic.
    """
    text: str
    descriptors: list[str] = field(default_factory=list)
    location: Optional[SpatialLocation] = None
    property_reference: Optional[str] = None
    property_reference_target: Optional[str] = None
    quantity: Optional[int] = 1      # 1=singular, None=all, N=explicit count
    timestamp: tuple = (0.0, 0.0)
    concrete: Optional[bool] = None

    # Deictic terms — if text is one of these, concrete is always False
    _DEICTIC = {"this","that","it","here","there","over there",
                "that one","this one","that thing","this thing"}

    def __post_init__(self):
        if isinstance(self.timestamp, list):
            self.timestamp = tuple(self.timestamp)
        if isinstance(self.location, dict):
            self.location = SpatialLocation.from_dict(self.location)
        # Fill in concrete if Phi-3 forgot to output it
        if self.concrete is None:
            self.concrete = self.text.lower().strip() not in self._DEICTIC

    # ------------------------------------------------------------------
    # Descriptor helpers
    # ------------------------------------------------------------------

    def has_descriptor(self, kind: str) -> bool:
        return any(kind.lower() in d.lower() for d in self.descriptors)

    def color_descriptors(self) -> list[str]:
        COLOR_KEYWORDS = {
            "red","green","blue","yellow","orange","purple","pink","brown",
            "black","white","grey","gray","cyan","magenta","beige","gold",
            "silver","dark","light","bright","pale","striped","spotted",
        }
        return [d for d in self.descriptors
                if any(c in d.lower() for c in COLOR_KEYWORDS)]

    def size_descriptors(self) -> list[str]:
        SIZE_KEYWORDS = {
            "large","small","big","tiny","huge","tall","short","wide",
            "narrow","thick","thin","long","fat","flat","slim",
        }
        return [d for d in self.descriptors
                if any(s in d.lower() for s in SIZE_KEYWORDS)]

    def shape_descriptors(self) -> list[str]:
        SHAPE_KEYWORDS = {
            "round","square","rectangular","cylindrical","circular","oval",
            "triangular","cubic","spherical","flat","curved","pointed",
        }
        return [d for d in self.descriptors
                if any(s in d.lower() for s in SHAPE_KEYWORDS)]

    def material_descriptors(self) -> list[str]:
        MATERIAL_KEYWORDS = {
            "wooden","plastic","metal","metallic","glass","ceramic","rubber",
            "fabric","cloth","paper","cardboard","stone","marble","leather",
        }
        return [d for d in self.descriptors
                if any(m in d.lower() for m in MATERIAL_KEYWORDS)]

    def detection_query(self) -> str:
        """
        Build a detection query string: descriptors + text.
        e.g. "small red mug"
        Location intentionally excluded — the detector works on crops.
        """
        parts = self.descriptors + ([self.text] if self.text else [])
        return " ".join(parts) if parts else ""


@dataclass
class ActionWord:
    """The action (verb / phrasal verb) extracted from the transcription."""
    text: str
    timestamp: tuple = (0.0, 0.0)

    def __post_init__(self):
        if isinstance(self.timestamp, list):
            self.timestamp = tuple(self.timestamp)


@dataclass
class Command:
    """A fully parsed pick-and-place command."""
    object: EntityWord
    action: ActionWord
    target: EntityWord

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_json_str(cls, raw: str) -> "Command":
        cleaned = (
            raw.replace("```json", "")
               .replace("```", "")
               .replace("(", "[")
               .replace(")", "]")
               .replace("None", "null")
               .strip()
        )
        data = json.loads(cleaned)

        def _parse_entity(d) -> EntityWord:
            # Phi-3 sometimes outputs a plain string instead of a dict
            if isinstance(d, str):
                return EntityWord(text=d, concrete=d.lower() not in EntityWord._DEICTIC)
            if not isinstance(d, dict):
                return EntityWord(text="", concrete=False)
            loc = d.get("location")
            return EntityWord(
                text=d.get("text", ""),
                descriptors=d.get("descriptors", []),
                location=SpatialLocation.from_dict(loc) if isinstance(loc, dict) else None,
                property_reference=d.get("property_reference"),
                property_reference_target=d.get("property_reference_target"),
                quantity=d.get("quantity", 1),
                timestamp=tuple(d.get("timestamp", [0.0, 0.0])),
                concrete=d.get("concrete"),
            )

        def _parse_action(a) -> ActionWord:
            # Phi-3 sometimes outputs a plain string instead of a dict
            if isinstance(a, str):
                return ActionWord(text=a)
            if not isinstance(a, dict):
                return ActionWord(text="")
            return ActionWord(
                text=a.get("text", ""),
                timestamp=tuple(a.get("timestamp", [0.0, 0.0])),
            )

        return cls(
            object=_parse_entity(data.get("object", {})),
            action=_parse_action(data.get("action", {})),
            target=_parse_entity(data.get("target", {})),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        def _entity_dict(e):
            return {
                "text": e.text,
                "descriptors": e.descriptors,
                "location": e.location.to_dict() if e.location else None,
                "property_reference": e.property_reference,
                "property_reference_target": e.property_reference_target,
                "quantity": e.quantity,
                "timestamp": list(e.timestamp),
                "concrete": e.concrete,
            }
        return {
            "object": _entity_dict(self.object),
            "action": {
                "text": self.action.text,
                "timestamp": list(self.action.timestamp),
            },
            "target": _entity_dict(self.target),
        }

    def to_json_str(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def save(self, path: Union[str, Path]):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json_str())

    # ------------------------------------------------------------------
    # Helpers for main.py
    # ------------------------------------------------------------------

    def is_fully_concrete(self) -> bool:
        return bool(self.object.concrete and self.target.concrete)

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.object.text:
            missing.append("object")
        if not self.action.text:
            missing.append("action")
        if not self.target.text:
            missing.append("target")
        return missing

    def object_matching_mode(self) -> str:
        """
        Infer which visual matching strategy main.py should use.
        Returns: "color" | "shape" | "size" | "position" | "none"
        Priority: color > shape > size > position > none.
        """
        if self.object.color_descriptors():
            return "color"
        if self.object.shape_descriptors():
            return "shape"
        if self.object.size_descriptors():
            return "size"
        if self.object.location:
            return "position"
        return "none"


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def _transcription_text(transcription) -> str:
    """Extract plain text from a Whisper transcription dict or return as-is."""
    if isinstance(transcription, dict):
        return transcription.get("text", str(transcription))
    return str(transcription)


class CommandExtractor:
    """
    Extracts a structured pick-and-place Command from a Whisper transcription.
    Single-pass. Outputs object/target with descriptors, structured location,
    timestamps, and concreteness.

    Supports multiple model families (phi3, qwen) — prompt set is selected
    automatically from the model_id. Add entries to PROMPTS to support more.
    """

    _DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

    def __init__(self, model_id: str = _DEFAULT_MODEL,
                 device: str = "cuda",
                 torch_dtype: str = "auto",
                 load_in_4bit: bool = False) -> None:

        self._model_id = model_id
        self._prompts = PROMPTS[_resolve_model_family(model_id)]

        load_kwargs = dict(
            trust_remote_code=True,
            device_map=device,
            torch_dtype=torch_dtype,
            attn_implementation="eager",   # avoids FlashAttentionKwargs on old transformers
        )
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype if torch_dtype != "auto" else "float16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            # device_map handles placement when quantising
            load_kwargs.pop("device_map", None)
            load_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    def _generate(self, messages: list[dict], max_new_tokens: int = 600) -> str:
        raw = self._pipe(
            messages,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            do_sample=False,
        )[0]["generated_text"]

        import re as _re
        cleaned = (
            raw.replace("```json", "")
               .replace("```", "")
               .replace("None", "null")
               .strip()
        )
        # Strip any preamble text — find the first JSON object or array
        m = _re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', cleaned)
        return m.group(0) if m else cleaned

    def _p(self, key: str) -> str:
        """Return the prompt string for the current model family."""
        return self._prompts[key]

    def extract(self, transcription: dict) -> Command:
        messages = [
            {"role": "system", "content": self._p("extract")},
            {"role": "user",   "content": str(transcription)},
        ]
        output = self._generate(messages)
        import logging as _logging
        _logging.getLogger("uncom_reasoner").debug("extract() raw output: %s", output)
        return Command.from_json_str(output)

    def detect_groups(self, transcription: dict) -> dict:
        """Pre-pass: detect whether the transcription refers to a group of objects.

        Returns a dict with keys:
            is_group  (bool)
            descriptor (str)   — singular noun, "" if purely deictic
            reference  (str|None) — anchor object name, None if gesture-only
        """
        import logging
        _log = logging.getLogger("uncom_reasoner")

        text = _transcription_text(transcription)
        messages = [
            {"role": "system", "content": self._p("group_detect")},
            {"role": "user",   "content": text},
        ]
        output = self._generate(messages, max_new_tokens=100)
        try:
            result = json.loads(output)
            return {
                "is_group":   bool(result.get("is_group", False)),
                "descriptor": result.get("descriptor") or "",
                "reference":  result.get("reference"),
            }
        except Exception as e:
            _log.warning("detect_groups parse failed (%s), assuming no group", e)
            return {"is_group": False, "descriptor": "", "reference": None}

    def extract_all(self, transcription: dict) -> list:
        """Extract one or more Commands from a Whisper transcription.

        Handles sequential commands, quantity (all/N/"these"), and incomplete
        commands (no object = place held; no target = pick and hold).

        Returns:
            List[Command]. Falls back to [self.extract()] on parse failure.
        """
        import logging
        _log = logging.getLogger("uncom_reasoner")

        # --- Pre-pass: group detection -----------------------------------
        # --- Pre-pass: group detection — only if text contains group triggers ---
        _GROUP_TRIGGERS = {
            "these", "those", "them", "all", "both", "every",
            "pile", "bunch", "stack", "group",
        }
        text_lower = _transcription_text(transcription).lower()
        if any(w in text_lower.split() for w in _GROUP_TRIGGERS):
            group_info = self.detect_groups(transcription)
            _log.info("detect_groups result: %s", group_info)
        else:
            group_info = {"is_group": False, "descriptor": None, "reference": None}
            _log.info("detect_groups skipped — no group trigger words found")

        messages = [
            {"role": "system", "content": self._p("extract_multi")},
            {"role": "user",   "content": str(transcription)},
        ]
        output = self._generate(messages, max_new_tokens=1200)

        try:
            cleaned = (
                output.replace("```json", "")
                      .replace("```", "")
                      .replace("(", "[")
                      .replace(")", "]")
                      .replace("None", "null")
                      .strip()
            )
            data = json.loads(cleaned)
            if not isinstance(data, list):
                data = [data]
            commands = [Command.from_json_str(json.dumps(d)) for d in data]
        except Exception as e:
            _log.warning("extract_all parse failed (%s), falling back to extract()", e)
            try:
                commands = [self.extract(transcription)]
            except Exception as e2:
                _log.error("extract() fallback also failed (%s), returning empty command", e2)
                commands = [Command(
                    object=EntityWord(text="", concrete=False),
                    action=ActionWord(text=""),
                    target=EntityWord(text="", concrete=False),
                )]

        # --- Post-parse cleanup: Phi-3 sometimes sets property_reference on the
        # object to the destination noun (e.g. "near bowl" when bowl is the target).
        # If property_reference_target matches the target text, it's a destination
        # not a spatial anchor — clear it so the cascade uses direct detection.
        for cmd in commands:
            prt = (cmd.object.property_reference_target or "").lower().strip()
            tgt = (cmd.target.text or "").lower().strip()
            if prt and tgt and prt == tgt:
                _log.info(
                    "Clearing spurious property_reference '%s'/'%s' — matches target",
                    cmd.object.property_reference, prt,
                )
                cmd.object.property_reference = None
                cmd.object.property_reference_target = None

        # --- Post-patch: apply group info to first pick command ----------
        # Only patch if the object is deictic/empty AND group was detected.
        # If Phi-3 already resolved a concrete noun, trust it and skip.
        if group_info["is_group"]:
            for cmd in commands:
                if cmd.object.text or cmd.object.quantity:
                    # Only override if the object is not already a concrete noun
                    if not cmd.object.concrete:
                        cmd.object.quantity = "these"
                        if group_info["descriptor"]:
                            cmd.object.text = group_info["descriptor"]
                    elif cmd.object.quantity in (None, "these"):
                        # Extractor already set quantity=these — just add anchor
                        cmd.object.quantity = "these"
                    else:
                        # Concrete noun with quantity=1 — group detect was wrong, skip
                        _log.info("group_detect fired but object '%s' is concrete/singular — skipping patch",
                                  cmd.object.text)
                        break
                    # Set reference anchor if given
                    if group_info["reference"]:
                        cmd.object.property_reference = "near"
                        cmd.object.property_reference_target = group_info["reference"]
                    break  # only patch the first pick command

        return commands

    def check_command_completeness(self, text: str) -> bool:
        messages = [
            {"role": "system", "content": self._p("complete")},
            {"role": "user",   "content": str(text)},
        ]
        output = self._generate(messages, max_new_tokens=10)
        return "true" in output.lower()

    def check_agreement(self, text: str) -> bool:
        messages = [
            {"role": "system", "content": self._p("agreement")},
            {"role": "user",   "content": str(text)},
        ]
        output = self._generate(messages, max_new_tokens=5).strip().replace("'", "")
        return output == "1"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def check_relative_position(text: str) -> Union[str, bool]:
    """
    Scan text for spatial keywords. Returns first match or False.
    NOTE: with the new schema prefer entity.location.relation directly —
    this is kept for call sites in main.py that normalise a location string
    to a single keyword for _find_object_by_relative_position().
    """
    SPATIAL_KEYWORDS = [
        "left", "right", "next", "beside", "between",
        "front", "behind", "near", "close", "above",
        "below", "under", "inside", "on top", "beneath",
        "across", "against", "along", "around",
    ]
    text_lower = text.lower()
    for kw in SPATIAL_KEYWORDS:
        if kw in text_lower:
            return kw
    return False


def check_agreement(transcription: str, device: str = "cuda",
                    model_id: str = "microsoft/Phi-3-mini-4k-instruct",
                    torch_dtype: str = "auto") -> str:
    """
    Standalone agreement check. Kept for main.py's check_agree().
    Reuses CommandExtractor so model loading is consistent (eager attn, same model).
    """
    extractor = CommandExtractor(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    messages = [
        {"role": "system", "content": extractor._p("agreement")},
        {"role": "user",   "content": str(transcription)},
    ]
    result = extractor._generate(messages, max_new_tokens=5)
    return result.replace("'", "").strip()