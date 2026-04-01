"""
Quick test for the Irodori-TTS API server.
  Section 1: one request per speaker in references_voices/
  Section 2: emoji-controlled synthesis (whisper, laugh, cry, etc.)

Start the server first:
  cd Irodori-TTS
  python api_server.py \
      --checkpoint Aratako/Irodori-TTS-500M-v2 \
      --devices cuda:0 \
      --precision bf16
"""

from __future__ import annotations

from pathlib import Path

import requests

API_URL = "http://localhost:8000"
VOICES_DIR = Path("references_voices")  # directory of per-speaker reference audio
OUTPUT_DIR = Path("test_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reference voice used for emoji tests — replace with any file from VOICES_DIR.
EMOJI_REF_VOICE = VOICES_DIR / "ナツメ" / "merged_audio.mp3"

BASE_TEXT = "今日はいい天気ですね。どこかお出かけしますか？"

# Emoji test cases: (label, text_with_emoji)
EMOJI_CASES = [
    ("whisper", f"👂{BASE_TEXT}"),
    ("giggle", f"🤭{BASE_TEXT}"),
    ("cry", f"😭{BASE_TEXT}"),
    ("angry", f"😠{BASE_TEXT}"),
    ("sleepy", f"😪{BASE_TEXT}"),
    ("fast", f"⏩{BASE_TEXT}"),
    ("slow", f"🐢{BASE_TEXT}"),
    ("phone", f"📞{BASE_TEXT}"),
    ("joyful", f"😆{BASE_TEXT}"),
    ("surprised", f"😲{BASE_TEXT}"),
]

# ---
#  Health check
# ---
resp = requests.get(f"{API_URL}/health", timeout=5)
resp.raise_for_status()
print("health:", resp.json())

# ------------------------------------------------------------------ #
#  Section 1: all speakers, standard text                             #
# ------------------------------------------------------------------ #
print("\n=== Section 1: speaker voice cloning ===")
speakers = sorted(VOICES_DIR.iterdir())
print(f"Found {len(speakers)} speakers: {[s.name for s in speakers]}\n")

for speaker_dir in speakers:
    ref_audio = speaker_dir / "merged_audio.mp3"
    if not ref_audio.exists():
        print(f"[skip] {speaker_dir.name}: merged_audio.mp3 not found")
        continue

    print(f"[{speaker_dir.name}] synthesizing …", end=" ", flush=True)
    with open(ref_audio, "rb") as f:
        response = requests.post(
            f"{API_URL}/synthesize",
            files={"reference_audio": ("merged_audio.mp3", f, "audio/mpeg")},
            data={
                "text": BASE_TEXT,
                "seconds": 10.0,
                "num_steps": 40,
                "cfg_scale_text": 3.0,
                "cfg_scale_speaker": 5.0,
            },
            timeout=120,
        )

    if response.status_code != 200:
        print(f"ERROR {response.status_code}: {response.text}")
        continue

    out_path = OUTPUT_DIR / f"{speaker_dir.name}.wav"
    out_path.write_bytes(response.content)
    print(f"saved → {out_path}  ({len(response.content) / 1024:.1f} KB)")

# ------------------------------------------------------------------ #
#  Section 2: emoji-controlled styles (fixed speaker: ナツメ)         #
# ------------------------------------------------------------------ #
print("\n=== Section 2: emoji styles ===")

if not EMOJI_REF_VOICE.exists():
    print(f"[skip] emoji tests: {EMOJI_REF_VOICE} not found")
else:
    emoji_out_dir = OUTPUT_DIR / "emoji"
    emoji_out_dir.mkdir(parents=True, exist_ok=True)

    for label, text in EMOJI_CASES:
        print(f"[{label}] {text!r} …", end=" ", flush=True)
        with open(EMOJI_REF_VOICE, "rb") as f:
            response = requests.post(
                f"{API_URL}/synthesize",
                files={"reference_audio": ("merged_audio.mp3", f, "audio/mpeg")},
                data={
                    "text": text,
                    "seconds": 10.0,
                    "num_steps": 40,
                    "cfg_scale_text": 3.0,
                    "cfg_scale_speaker": 5.0,
                },
                timeout=120,
            )

        if response.status_code != 200:
            print(f"ERROR {response.status_code}: {response.text}")
            continue

        out_path = emoji_out_dir / f"{label}.wav"
        out_path.write_bytes(response.content)
        print(f"saved → {out_path}  ({len(response.content) / 1024:.1f} KB)")

print("\nDone.")
