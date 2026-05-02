"""
Prompt Loader
==============
Scans the `prompts/` folder at runtime and builds a registry of all
available prompt variants.  Adding a new prompt style requires only
dropping a new .py file in that folder — no other file needs editing.

REQUIRED INTERFACE (each prompt module must define):
    NAME          str   — display label for the UI dropdown
    DESCRIPTION   str   — one-line tooltip
    SYSTEM_PROMPT str   — system-role text sent to the VLM
    FEW_SHOT      str   — few-shot examples appended before the user turn

USAGE:
    from prompt_loader import list_prompt_files, load_prompt, get_prompt_registry

    # Get all available prompt file stems (e.g. ["standard", "benchmark"])
    stems = list_prompt_files()

    # Load one prompt by file stem
    p = load_prompt("standard")
    print(p["system"])   # SYSTEM_PROMPT
    print(p["few_shot"]) # FEW_SHOT
    print(p["name"])     # NAME
    print(p["description"])

    # Full registry dict  {stem: {name, description, system, few_shot}}
    registry = get_prompt_registry()
"""

import importlib
import importlib.util
import os
import sys
from typing import Dict, List

# Absolute path to the prompts/ folder, resolved relative to this file
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Prompts")

# Required attributes every prompt module must export
_REQUIRED = ("NAME", "DESCRIPTION", "SYSTEM_PROMPT", "FEW_SHOT")


def list_prompt_files() -> List[str]:
    """
    Return sorted list of prompt file stems found in prompts/.
    Example: ["benchmark", "standard"]
    """
    if not os.path.isdir(PROMPTS_DIR):
        return []
    stems = []
    for fname in sorted(os.listdir(PROMPTS_DIR)):
        if fname.startswith("_") or not fname.endswith(".py"):
            continue
        stems.append(fname[:-3])  # strip .py
    return stems


def load_prompt(stem: str) -> Dict[str, str]:
    """
    Dynamically import prompts/<stem>.py and return a normalised dict:
        {
            "stem":        str,  # file stem, used as internal key
            "name":        str,  # NAME attribute
            "description": str,  # DESCRIPTION attribute
            "system":      str,  # SYSTEM_PROMPT attribute
            "few_shot":    str,  # FEW_SHOT attribute
        }

    Raises FileNotFoundError if the file doesn't exist.
    Raises AttributeError  if required attributes are missing.
    """
    fpath = os.path.join(PROMPTS_DIR, f"{stem}.py")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(
            f"Prompt file not found: {fpath}\n"
            f"Available stems: {list_prompt_files()}"
        )

    # Use a unique module name so re-loads don't collide
    mod_name = f"_agentvigil_prompt_{stem}"
    spec = importlib.util.spec_from_file_location(mod_name, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    # Validate required attributes
    missing = [attr for attr in _REQUIRED if not hasattr(mod, attr)]
    if missing:
        raise AttributeError(
            f"Prompt file '{stem}.py' is missing required attributes: {missing}\n"
            f"Each prompt module must define: {list(_REQUIRED)}"
        )

    return {
        "stem":        stem,
        "name":        mod.NAME,
        "description": mod.DESCRIPTION,
        "system":      mod.SYSTEM_PROMPT,
        "few_shot":    mod.FEW_SHOT,
    }


def get_prompt_registry() -> Dict[str, Dict[str, str]]:
    """
    Scan prompts/ and return a dict of all valid prompt variants.

        {stem: {stem, name, description, system, few_shot}, ...}

    Invalid or broken files are skipped with a warning (non-fatal).
    """
    registry: Dict[str, Dict[str, str]] = {}
    for stem in list_prompt_files():
        try:
            registry[stem] = load_prompt(stem)
        except (AttributeError, Exception) as exc:
            print(f"[prompt_loader] WARNING: skipping '{stem}.py' — {exc}")
    return registry


def get_default_stem() -> str:
    """
    Return the stem to use when nothing is selected yet.
    Prefers 'standard' if available, otherwise the first file found.
    """
    stems = list_prompt_files()
    if not stems:
        raise RuntimeError(
            f"No prompt files found in {PROMPTS_DIR}. "
            "Add at least one .py file with NAME, DESCRIPTION, SYSTEM_PROMPT, FEW_SHOT."
        )
    return "standard" if "standard" in stems else stems[0]