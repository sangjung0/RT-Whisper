# pip install -U transformers huggingface_hub accelerate sentencepiece
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# large-v3
m1 = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")
p1 = AutoProcessor.from_pretrained("openai/whisper-large-v3")

# large-v2
m2 = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")
p2 = AutoProcessor.from_pretrained("openai/whisper-large-v2")
