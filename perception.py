import re
from typing import List, Tuple

from config import VLM_REGISTRY, DEFAULT_VLM
from adapters import ADAPTER_CLASSES


# ══════════════════════════════════════════════════════════════════════
# EVENT EXTRACTION  (shared utility — no model dependency)
# ══════════════════════════════════════════════════════════════════════

# Canonical category aliases for normalisation
CANONICAL_CATEGORIES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
    "Normal_Videos_event",
]

CATEGORY_NORMALIZE = {
    "abuse": "Abuse", "arrest": "Arrest", "arson": "Arson",
    "assault": "Assault", "burglary": "Burglary",
    "explosion": "Explosion", "fighting": "Fighting",
    "roadaccidents": "RoadAccidents", "robbery": "Robbery",
    "shooting": "Shooting", "shoplifting": "Shoplifting",
    "stealing": "Stealing", "vandalism": "Vandalism",
    "normal_videos_event": "Normal_Videos_event",
    "normal": "Normal_Videos_event",
    "road accident": "RoadAccidents", "road accidents": "RoadAccidents",
    "road accident / vehicle collision": "RoadAccidents",
    "vehicle collision": "RoadAccidents",
    "robbery / armed robbery": "Robbery", "armed robbery": "Robbery",
    "burglary / breaking and entering": "Burglary",
    "breaking and entering": "Burglary",
    "vandalism / property damage": "Vandalism",
    "property damage": "Vandalism",
    "arson / fire": "Arson", "fire": "Arson",
    "normal activity": "Normal_Videos_event",
    "normal_videos": "Normal_Videos_event",
    "normal videos event": "Normal_Videos_event",
    "normal video": "Normal_Videos_event",
}


def _resolve_compound_category(raw: str) -> str:
    """
    Resolve labels like "Shoplifting / Stealing" without hardcoded
    cross-category collapse.

    Strategy: split by separators and return the first token that maps
    cleanly to a canonical category.
    """
    pieces = re.split(r"\s*(?:/|,|\bor\b|\||;)\s*", raw.strip(), flags=re.IGNORECASE)
    for piece in pieces:
        part = piece.strip().lower()
        if not part:
            continue
        if part in CATEGORY_NORMALIZE:
            mapped = CATEGORY_NORMALIZE[part]
            if mapped in CANONICAL_CATEGORIES:
                return mapped
        for canon in CANONICAL_CATEGORIES:
            if part == canon.lower():
                return canon
    return ""


def normalize_category(raw: str) -> str:
    if not raw:
        return "Normal_Videos_event"
    clean = raw.strip().lower()

    # Handle compound labels first to avoid accidental cross-category aliases.
    if any(sep in clean for sep in ["/", "|", ",", ";"]) or re.search(r"\bor\b", clean):
        resolved = _resolve_compound_category(clean)
        if resolved:
            return resolved

    if clean in CATEGORY_NORMALIZE:
        return CATEGORY_NORMALIZE[clean]
    clean2 = re.sub(r"[/_\-]", " ", clean).strip()
    if clean2 in CATEGORY_NORMALIZE:
        return CATEGORY_NORMALIZE[clean2]
    for canon in CANONICAL_CATEGORIES:
        if clean.startswith(canon.lower()):
            return canon
    return raw


def extract_event_from_caption(caption: str) -> str:
    """
    Parse the EVENT: <category> tag the prompt instructs the model to emit.

    Uses findall + last match to avoid picking up few-shot example tags
    that appear earlier in the string.
    """
    patterns = [
        r"EVENT:\s*([A-Za-z_/\s\-]+?)(?:\.|$|\n|\"|\')",
        r"Event:\s*([A-Za-z_/\s\-]+?)(?:\.|$|\n|\"|\')",
        r"event:\s*([A-Za-z_/\s\-]+?)(?:\.|$|\n|\"|\')",
        r"Classification:\s*([A-Za-z_/\s\-]+?)(?:\.|$|\n|\"|\')",
    ]
    for p in patterns:
        matches = re.findall(p, caption)
        if matches:
            tag = matches[-1].strip().rstrip(".")
            canon = normalize_category(tag)
            if canon in CANONICAL_CATEGORIES:
                return canon
    return "Normal_Videos_event"

def extract_structured_output(caption: str) -> dict:
    """
    Parse Phase-2 analyst-style VLM output into a structured dict.

    The prompt asks the VLM to emit:
        Detailed: ...
        Summary: ...
        CONFIRMED: <yes|no>
        EVIDENCE: <2-3 sentences>
        SEVERITY: <low|medium|high>
        EVENT: <category>

    Returns:
        {
            "event":     str   — canonical category (always set, default Normal_Videos_event)
            "confirmed": bool  — whether VLM confirmed the gate's hypothesis (default False)
            "severity":  str   — "low" | "medium" | "high" (default "low")
            "evidence":  str   — VLM's evidence text, truncated to 500 chars
        }

    Backward compatible — if the VLM emits only the old Detailed/Summary/EVENT
    format, CONFIRMED/SEVERITY/EVIDENCE default to False/low/"" and EVENT is
    still parsed correctly via extract_event_from_caption.
    """
    result = {
        "event":     "Normal_Videos_event",
        "confirmed": False,
        "severity":  "low",
        "evidence":  "",
    }

    # EVENT — delegate to existing parser
    result["event"] = extract_event_from_caption(caption)

    # CONFIRMED line
    m = re.search(r"CONFIRMED:\s*(yes|no|true|false)\b", caption, re.IGNORECASE)
    if m:
        result["confirmed"] = m.group(1).lower() in ("yes", "true")

    # SEVERITY line
    m = re.search(r"SEVERITY:\s*(low|medium|high)\b", caption, re.IGNORECASE)
    if m:
        result["severity"] = m.group(1).lower()

    # EVIDENCE — capture until the next known label or end of string
    m = re.search(
        r"EVIDENCE:\s*(.+?)(?:\n\s*(?:SEVERITY|EVENT|CONFIRMED|SUMMARY):|\Z)",
        caption, re.IGNORECASE | re.DOTALL,
    )
    if m:
        result["evidence"] = m.group(1).strip()[:500]

    return result

# ══════════════════════════════════════════════════════════════════════
# PERCEPTION ENGINE — public facade used by app.py
# ══════════════════════════════════════════════════════════════════════

class PerceptionEngine:
    """
    Thin facade over the selected VLM adapter.

    app.py calls:
        perception.caption_chunk(frame_paths, prompt_type)
        perception.extract_event_type(caption)
        perception.unload()

    Switching models requires constructing a new PerceptionEngine with
    a different model_key.
    """

    def __init__(self, model_key: str = DEFAULT_VLM):
        self.model_key = model_key
        cfg = VLM_REGISTRY.get(model_key)
        if cfg is None:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Available: {list(VLM_REGISTRY.keys())}"
            )
        adapter_cls = ADAPTER_CLASSES.get(cfg["adapter"])
        if adapter_cls is None:
            raise ValueError(
                f"Unknown adapter '{cfg['adapter']}' for model '{model_key}'."
            )
        from adapters.base import BaseVideoAdapter
        self._adapter: BaseVideoAdapter = adapter_cls(cfg["model_id"])
        self.display_name = cfg["display_name"]

    def load(self) -> None:
        self._adapter.load()

    def caption_chunk(
            self,
            frame_paths: List[str],
            prompt_type: str = "standard",
            prev_context: str = "",
            flashback_prior: str = "",
        ) -> Tuple[str, float]:
            return self._adapter.caption_chunk(
                frame_paths, prompt_type, prev_context, flashback_prior,
            )

    def extract_event_type(self, caption: str) -> str:
        return extract_event_from_caption(caption)
    
    def extract_structured_output(self, caption: str) -> dict:
        return extract_structured_output(caption)

    def unload(self) -> None:
        self._adapter.unload()

    @property
    def loaded(self) -> bool:
        return self._adapter.loaded
