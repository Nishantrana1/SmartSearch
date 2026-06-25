# searcher.py
# runs CLIP on extracted frames to find the search query
# uses zero-shot classification (softmax over contrastive prompts)
# model loads once via init(), all inference runs under torch.no_grad()

import clip  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
from PIL import Image  # type: ignore

# module-level state — populated by init()
DEVICE     = None
MODEL      = None
PREPROCESS = None

# softmax probability threshold — how confident the zero-shot classifier
# must be that the query matches. Range 0.0 to 1.0.
# 0.55 means CLIP must assign >55% probability to the query over the
# negative prompts — a solid, discriminative threshold.
CONFIDENCE_THRESHOLD = 0.55

# how many images to encode at once (higher = faster but uses more RAM)
BATCH_SIZE = 16

# negative (contrastive) prompts used to distinguish real matches
# from background noise. CLIP compares the query against these to decide.
NEGATIVE_PROMPTS = [
    "a photo of a random scene",
    "a photo of nothing in particular",
    "a blurry photo",
]


def init():
    """
    Load the CLIP model once. Call this at startup from main.py.
    Subsequent calls are no-ops.
    """
    global DEVICE, MODEL, PREPROCESS

    if MODEL is not None:
        return  # already loaded

    print("Loading CLIP model...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
    MODEL.eval()
    print(f"CLIP ready on {DEVICE}")


def search_single_video(video_path, frames, query):
    """
    Search one video's frames for the query using zero-shot classification.
    Compares the query prompt against negative prompts via softmax to get
    a true probability, not just a raw cosine score.
    Returns a result dict if found, None if not found.
    """
    if not frames:
        return None

    with torch.no_grad():
        # build the full set of prompts: query + negatives
        query_prompt = f"a photo of {query}"
        all_prompts  = [query_prompt] + NEGATIVE_PROMPTS

        # encode all text prompts at once
        text_tokens   = clip.tokenize(all_prompts).to(DEVICE)
        text_features = MODEL.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        matches = []

        # process frames in batches for speed
        for batch_start in range(0, len(frames), BATCH_SIZE):
            batch = frames[batch_start : batch_start + BATCH_SIZE]

            # preprocess all images in this batch
            image_tensors = []
            valid_indices  = []
            for i, frame_data in enumerate(batch):
                try:
                    # convert opencv frame (BGR numpy array) to PIL image (RGB)
                    # .copy() ensures contiguous memory for PIL
                    image_rgb = frame_data["image"][:, :, ::-1].copy()
                    pil_image = Image.fromarray(image_rgb)
                    image_tensors.append(PREPROCESS(pil_image))
                    valid_indices.append(i)
                except Exception:
                    # skip frames that fail to convert
                    continue

            if not image_tensors:
                continue

            # stack into a single batch tensor and encode
            image_batch    = torch.stack(image_tensors).to(DEVICE)
            image_features = MODEL.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # compute similarities against ALL prompts: shape (N_images, N_prompts)
            # scale by CLIP's learned temperature (logit_scale) for proper softmax
            logit_scale  = MODEL.logit_scale.exp()
            logits       = logit_scale * (image_features @ text_features.T)

            # softmax across prompts → probability that each prompt matches
            probabilities = logits.softmax(dim=-1)

            # column 0 is the query prompt probability
            query_probs = probabilities[:, 0]
            query_probs = torch.atleast_1d(query_probs)

            # check each frame in the batch
            for j, prob_tensor in enumerate(query_probs):
                prob = prob_tensor.item()
                if prob >= CONFIDENCE_THRESHOLD:
                    frame_data = batch[valid_indices[j]]
                    matches.append({
                        "timestamp": frame_data["timestamp"],
                        "seconds":   frame_data["seconds"],
                        "score":     prob
                    })

    if not matches:
        return None

    # sort matches by timestamp
    matches.sort(key=lambda x: x["seconds"])

    # remove duplicate timestamps that are too close together (within 2 seconds)
    filtered = [matches[0]]
    for m in matches[1:]:
        if m["seconds"] - filtered[-1]["seconds"] > 2.0:
            filtered.append(m)

    return {
        "video":      video_path,
        "timestamps": [m["timestamp"] for m in filtered],
        "best_score": max(m["score"] for m in filtered),
        "matches":    filtered
    }