# debug_scores.py — prints softmax probabilities from the new zero-shot approach
from pathlib import Path
from extractor import extract_frames
import searcher
import clip, torch
from PIL import Image

searcher.init()

folder = Path(r"D:\Nisha\programing\project\SmartSearch\Random Video folder")
videos = list(folder.glob("*"))
videos = [v for v in videos if v.suffix.lower() in {".mp4",".avi",".mkv",".mov",".wmv",".flv",".webm"}]

query = "car"
query_prompt = f"a photo of {query}"
all_prompts = [query_prompt] + searcher.NEGATIVE_PROMPTS

print(f"\nQuery: '{query}'")
print(f"Prompts: {all_prompts}")
print(f"Threshold: {searcher.CONFIDENCE_THRESHOLD}")
print("=" * 70)

with torch.no_grad():
    text_tokens = clip.tokenize(all_prompts).to(searcher.DEVICE)
    text_features = searcher.MODEL.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = searcher.MODEL.logit_scale.exp()

    for vp in videos:
        print(f"\n--- {vp.name} ---")
        frames = extract_frames(vp, mode="fast")

        for i, fd in enumerate(frames):
            image_rgb = fd["image"][:, :, ::-1].copy()
            pil = Image.fromarray(image_rgb)
            img_t = searcher.PREPROCESS(pil).unsqueeze(0).to(searcher.DEVICE)
            img_f = searcher.MODEL.encode_image(img_t)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)

            logits = logit_scale * (img_f @ text_features.T)
            probs = logits.softmax(dim=-1).squeeze(0)

            query_prob = probs[0].item()
            flag = " <<<< MATCH" if query_prob >= searcher.CONFIDENCE_THRESHOLD else ""
            print(f"  frame {i:3d} @ {fd['timestamp']:>6s}  prob={query_prob:.4f}{flag}")
