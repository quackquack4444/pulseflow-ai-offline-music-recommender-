"""Generate natural‑language explanations for playlist transitions via Ollama.

This module provides a helper to call a local large language model via
Ollama’s HTTP API.  If the Ollama service is not running or a model is
not configured, the `explain_transition` function will return a
placeholder string.  See `config.yaml` for LLM settings.
"""

"""LLM explanations (Ollama/local)."""
#from __future__ import annotations

import os
import json
import time
from typing import Optional, List, Dict
import subprocess
import requests

import sys
from pathlib import Path

import yaml

try:
    import requests
except ImportError:
    print("The requests library is required for LLM explanations. Install it via pip install requests.", file=sys.stderr)
    raise


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def call_ollama(prompt: str, model: str, max_tokens: int, temperature: float, stop: Optional[list] = None) -> str:
    """Send a prompt to the local Ollama server and return the response text.

    If the server is unreachable or returns an error, raise an exception.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Ollama: {e}")
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama API error: {resp.status_code} {resp.text}")
    # The response may be streamed line‑by‑line; handle both streaming and full JSON
    try:
        data = resp.json()
        return data.get("response", "").strip()
    except json.JSONDecodeError:
        # If streaming, accumulate response
        text = ""
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
                delta = obj.get("response", "")
                text += delta
            except json.JSONDecodeError:
                pass
        return text.strip()


def explain_transition(
    current_track: Dict,
    next_track: Dict,
    metrics: Dict[str, float],
    policy: str,
    config: dict,
) -> str:
    """Generate a natural‑language explanation for the transition from current_track to next_track.

    Parameters:
        current_track: dict containing 'title', 'artist' and any descriptors
        next_track: dict containing 'title', 'artist' and descriptors
        metrics: dict with computed differences (e.g., style_distance, mood_delta, bpm_diff, key_diff)
        policy: either 'smooth' or 'contrast'
        config: configuration dictionary (must contain 'llm' section)

    Returns:
        Explanation string.  If Ollama is not configured, returns a simple
        description string instead.
    """
    llm_cfg = config.get("llm", {})
    model = llm_cfg.get("model")
    if not model:
        # Fallback explanation without LLM
        desc = []
        desc.append(f"Transitioning to {next_track['title']} by {next_track['artist']}.")
        desc.append(f"Style difference: {metrics['style_distance']:.2f}.")
        desc.append(f"Mood change: {metrics['mood_delta']:.2f}.")
        desc.append(f"BPM: {metrics['bpm_from']}→{metrics['bpm_to']}, key diff: {metrics['key_diff']}.")
        return " ".join(desc)
    # Compose prompt
    system_msg = "You are a concise music adviser (max 6 sentences)."
    user_msg = (
        f"Current track:\n  title: \"{current_track['title']}\"\n  artist: \"{current_track['artist']}\"\n"
        f"Next candidate:\n  title: \"{next_track['title']}\"\n  artist: \"{next_track['artist']}\"\n"
        f"Stats: style_distance={metrics['style_distance']:.2f}, mood_delta={metrics['mood_delta']:.2f}, "
        f"bpm: {metrics['bpm_from']}→{metrics['bpm_to']}, key_diff: {metrics['key_diff']}, energy_delta={metrics['energy_delta']:.2f}.\n"
        f"Explain why this is a good {policy} transition for focus. End with one emoji. </END>"
    )
    prompt = f"SYSTEM: {system_msg}\nUSER: {user_msg}"
    max_tokens = int(llm_cfg.get("max_tokens", 160))
    temperature = float(llm_cfg.get("temperature", 0.7))
    stop = llm_cfg.get("stop", None)
    try:
        response = call_ollama(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return response
    except Exception as e:
        # If LLM call fails, fall back
        print(f"LLM explanation failed: {e}", file=sys.stderr)
        desc = []
        desc.append(f"Transitioning to {next_track['title']} by {next_track['artist']}.")
        desc.append(f"Style difference: {metrics['style_distance']:.2f}.")
        desc.append(f"Mood change: {metrics['mood_delta']:.2f}.")
        desc.append(f"BPM: {metrics['bpm_from']}→{metrics['bpm_to']}, key diff: {metrics['key_diff']}.")
        return " ".join(desc)



def _ollama(
    prompt: str,
    model: str = "qwen2.5:3b",
    max_tokens: int = 160,
    temperature: float = 0.7,
    timeout: int = 90,
) -> str:
    """Generate text through the local Ollama API."""

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=(5, timeout),
        )
        response.raise_for_status()

        text = response.json().get("response", "").strip()

        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        return text

    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Ollama did not respond within {timeout} seconds."
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Ollama request failed: {exc}")

def explain_playlist_summary(
    tracks: List[Dict],
    policy: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> Dict[str, str]:

    titles = [
        str(t.get("title") or t.get("path") or "Unknown track")
        for t in tracks
    ]

    if not titles:
        return {"top_pick": "", "overview": "No tracks in playlist."}

    prompt = (
        "You are a concise music adviser. In 3 short sentences:\n"
        "1) Name ONE standout track from this list as 'Top pick'.\n"
        "2) Describe the overall vibe in 1 sentence.\n"
        f"3) Mention how the sequencing fits a {policy} policy.\n"
        "Tracks:\n- " + "\n- ".join(titles) + "\n"
        "Respond as:\n"
        "Top pick: <title>\n"
        "Overview: <one sentence>\n"
    )

    try:
        txt = _ollama(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        top, overview = "", ""

        for line in txt.splitlines():
            lower = line.lower()

            if lower.startswith("top pick"):
                top = line.split(":", 1)[-1].strip()
            elif lower.startswith("overview"):
                overview = line.split(":", 1)[-1].strip()

        return {
            "top_pick": top or titles[0],
            "overview": overview or txt[:240],
        }

    except Exception as exc:
        print(f"Ollama summary fallback: {exc}", flush=True)

        return {
            "top_pick": titles[0],
            "overview": (
                f"A {policy} playlist containing {len(titles)} tracks. "
                "The sequence was generated using the playlist ranking system."
            ),
        }

def explain_transition(
    curr: Dict,
    nxt: Dict,
    stats: Dict[str, str],
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Explain a transition, falling back to a basic description."""

    t1 = curr.get("title") or curr.get("path") or "Unknown track"
    t2 = nxt.get("title") or nxt.get("path") or "Unknown track"

    stats = stats or {}
    stat_text = ", ".join(
        f"{key}={value}" for key, value in stats.items()
    )

    prompt = (
        "Be concise (no more than 2 sentences). "
        "Explain why this transition works for listening.\n"
        f"From: {t1}\n"
        f"To: {t2}\n"
        f"Stats: {stat_text}\n"
        "End with one emoji."
    )

    try:
        return _ollama(
            prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    except Exception as exc:
        print(f"Ollama transition fallback: {exc}", flush=True)

        description = f"Moving from {t1} to {t2}."
        if stat_text:
            description += f" Available transition information: {stat_text}."

        return description

