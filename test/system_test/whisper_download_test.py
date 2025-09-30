import os
from faster_whisper import WhisperModel

WORKDIR = os.environ["CONTAINER_WORK_DIR"]
TEST_AUDIO_FILE = f"{WORKDIR}/.data/news_with_english.mp3"

model = WhisperModel("large-v3", device="cuda", compute_type="int8")

segments, info = model.transcribe(TEST_AUDIO_FILE, beam_size=5)

print(
    "Detected language '%s' with probability %f"
    % (info.language, info.language_probability)
)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
