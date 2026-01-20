import os
import json
import math
from typing import List

from openai import OpenAI

# ВАЖНО: этот скрипт импортирует KB_INDEX из main.py
# Поэтому в main.py load_kb() должен работать без запуска сервера.
from main import load_kb, KB_INDEX

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OUT_PATH = "knowledge/embeddings.json"
MODEL = "text-embedding-3-small"  # мультиязычная, RU<->EN работает

def batched(lst: List[str], n: int = 128):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("No OPENAI_API_KEY in env")

    n, msg = load_kb()
    print("KB loaded:", n, msg)

    # Берём тот же текст, по которому ты уже ищешь (it["text"])
    texts = []
    for it in KB_INDEX:
        t = (it.get("text") or "").strip()
        if not t:
            t = "empty"
        texts.append(t)

    client = OpenAI(api_key=OPENAI_API_KEY)

    vectors = []
    for chunk in batched(texts, 128):
        r = client.embeddings.create(model=MODEL, input=chunk)
        vectors.extend([d.embedding for d in r.data])
        print("Embedded:", len(vectors), "/", len(texts))

    os.makedirs("knowledge", exist_ok=True)
    payload = {
        "model": MODEL,
        "count": len(vectors),
        "dim": len(vectors[0]) if vectors else 0,
        # сохраняем по индексу: vectors[i] соответствует KB_INDEX[i]
        "vectors": vectors,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print("OK saved:", OUT_PATH)

if __name__ == "__main__":
    main()
