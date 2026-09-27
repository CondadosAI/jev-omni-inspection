"""A System One readout in Python, two ways, on the same photo and question.

1. Jev-Omni: a model trained for it. Its own predict() returns one probability per option.
2. Gemma 4 12B, untouched: one forward pass, then read the probability that the next token is
   "1" or "2". No training, no text generated.

  python examples/system_one_readout.py cashew_anomaly_000.jpg

Needs a CUDA GPU with about 80 GB free for both models (Jev-Omni's weights are FP32).
"""
import sys

import torch
from PIL import Image

PHOTO = sys.argv[1] if len(sys.argv) > 1 else "cashew_anomaly_000.jpg"
STATE = "Production-line inspection photo of a cashew nut."
QUESTION = "Is everything in the photo good, or is at least one part defective?"
OPTIONS = ["all good", "defective"]


def jev_omni(photo):
    from huggingface_hub import snapshot_download
    path = snapshot_download("akhilaaa3/Jev-Omni", revision="5addda86ddee081a68fb067477ea100c221b8917")
    sys.path.insert(0, path)
    from jev_omni import load_jev_omni

    classifier = load_jev_omni()
    result = classifier.predict(state=STATE, question=QUESTION, options=OPTIONS,
                                media=photo, modality="image")
    return result["probabilities"]


def gemma4_one_pass(photo):
    from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration

    model_id, revision = "google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, device_map="cuda")

    # The same prompt Jev-Omni builds: the options as numbers, and a request for the number only.
    numbered = "\n".join(f"{i + 1}. {o}" for i, o in enumerate(OPTIONS))
    prompt = (f"{STATE}\n\n---\n\nQUESTION: {QUESTION}\n\nOPTIONS:\n{numbered}\n\n"
              f"Reply with only the number of the correct option (1-{len(OPTIONS)}).\n"
              "Output a single number and nothing else.")
    messages = [{"role": "user", "content": [{"type": "image", "image": Image.open(photo).convert("RGB")},
                                             {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt", enable_thinking=False)
    inputs = inputs.to("cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        logits = model(**inputs, logits_to_keep=1).logits[0, -1]  # one pass, nothing generated

    digit_ids = [processor.tokenizer.convert_tokens_to_ids(str(i + 1)) for i in range(len(OPTIONS))]
    probs = torch.softmax(logits[digit_ids].float(), dim=0)
    return dict(zip(OPTIONS, probs.tolist()))


if __name__ == "__main__":
    which = sys.argv[2] if len(sys.argv) > 2 else "both"
    if which in ("both", "jev"):
        print("Jev-Omni:", {k: round(v, 3) for k, v in jev_omni(PHOTO).items()})
        torch.cuda.empty_cache()
    if which in ("both", "gemma"):
        print("Gemma 4, one pass:", {k: round(v, 3) for k, v in gemma4_one_pass(PHOTO).items()})
