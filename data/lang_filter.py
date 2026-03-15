import json

from matplotlib import pyplot as plt
from tqdm import tqdm
import re as re
import math
import multiprocessing
from tqdm.contrib.concurrent import thread_map as tqdm_thread_map

english_words = [
    "the", "of", "and", "to", "in", "is", "you", "that", "it", "he",
    "was", "for", "on", "are", "with", "as", "I", "his", "they", "be",
    "at", "one", "have", "this", "from", "by", "not", "but", "or", "an"
]
french_words = [
    "le", "la", "et", "un", "une", "des", "en", "du", "il", "elle",
    "ce", "qui", "que", "dans", "sur", "pour", "pas", "par", "mais", "aux",
    "au", "ne", "se", "est", "son", "ses", "ces", "mon", "ton", "nous",
    "été", "être", "où", "ça"
]
german_words = [
    "der", "die", "das", "und", "in", "zu", "von", "mit", "ist", "auf",
    "nicht", "ein", "eine", "es", "ich", "er", "sie", "wir", "ihr", "als",
    "auch", "für", "dem", "den", "nach", "bei", "noch", "wie", "über",
    "schön", "straße", "über", "für", "können"
]
spanish_words = [
    "el", "la", "los", "las", "un", "una", "y", "o", "en", "de",
    "que", "con", "por", "para", "es", "se", "del", "al", "como",
    "su", "sus", "te", "le", "lo", "sí", "mi", "tú", "él",
    "años", "niño", "mañana", "está", "corazón"
]
italian_words = [
    "il", "la", "e", "un", "una", "di", "che", "per", "non",
    "con", "si", "del", "al", "lo", "su", "mi", "ti", "ci",
    "vi", "gli", "dei", "da", "al", "ma", "o", "lui", "loro", "questo",
    "città", "perché", "questo", "dove", "anche"
]
dutch_words = [
    "de", "het", "en", "een", "van", "op", "te", "dat",
    "die", "niet", "met", "voor", "als", "ook", "zijn", "aan", "bij",
    "door", "maar", "hij", "zij", "wij", "jullie", "hun", "dit"
]
portuguese_words = [
    "o", "e", "um", "uma", "de", "em", "que", "não",
    "por", "para", "se", "dos", "da", "das", "os", "as", "eu",
    "tu", "ele", "ela", "nós", "vós", "eles", "elas", "te", "se",
    "coração", "país", "está", "também", "pão"
]
characteristic_words = {
    "English": english_words,
    "French": french_words,
    "German": german_words,
    "Spanish": spanish_words,
    "Italian": italian_words,
    "Dutch": dutch_words,
    "Portuguese": portuguese_words
}
lang_patterns = {
    key: re.compile(r"\b(?:" + "|".join(value) + r")\b", re.IGNORECASE)
    for key, value in characteristic_words.items()
}


def ascii_ratio(text: str) -> float:
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text)


def filter_worker(args: tuple[list[str], int, dict, bool]):
    filtered_chunk, thread_idx, patterns, show_progress = args
    probs = []
    for i in tqdm(filtered_chunk) if (show_progress and thread_idx == 0) else filtered_chunk:
        text_len = len(i)
        if text_len < 4096:
            sample = i[:2048]
        else:
            sample = i[:1024] + i[text_len // 2:text_len // 2 + 1024]
        matches = {
            key: sum(1 for _ in p.finditer(sample)) for key, p in patterns.items()
        }
        scale = max(len(sample) / 100, 1e-6)
        scores = {k: v / scale for k, v in matches.items()}
        scores["Uncertain"] = 1.0

        # softmax
        m = max(scores.values())
        exp_scores = {k: math.exp(v - m) for k, v in scores.items()}
        total = sum(exp_scores.values())
        prob = {k: v / total for k, v in exp_scores.items()}
        sorted_prob = dict(
            sorted(prob.items(), key=lambda x: x[1], reverse=True)
        )
        probs.append(sorted_prob)
    return probs


def filter(texts: list[str], plot: bool = False, show_progress: bool = True, ascii_bar: float = 0.98,
           lang_bar: float = 0.60, n_proc: int = 4) -> list[str]:
    ratios = []
    for i in tqdm(texts):
        ratios.append(ascii_ratio(i[:1024]))

    if plot:
        plt.hist(ratios, bins=100, range=(0, 1), edgecolor='black', log=True)
        plt.show()
        plt.hist(ratios, bins=100, range=(0, 1), edgecolor='black')
        plt.show()

    filtered = [i for i, ratio in zip(texts, ratios) if ratio > ascii_bar]

    mp_args = [
        (
            filtered[
                len(filtered) * i // n_proc:
                len(filtered) * (i + 1) // n_proc
            ],
            i,
            lang_patterns,
            show_progress
        )
        for i in range(n_proc)
    ]

    # probs_mt = [filter_worker((filtered, 0))]
    with multiprocessing.Pool(n_proc) as pool:
        probs_mp = pool.map(filter_worker, mp_args)

    probs = []
    for i in probs_mp:
        probs.extend(i)

    english_confs = [conf['English'] for conf in probs]
    if plot:
        plt.hist(english_confs, bins=100, range=(0, 1), edgecolor='black', log=True)
        plt.show()
        plt.hist(english_confs, bins=100, range=(0, 1), edgecolor='black')
        plt.show()
    # for text, conf in zip(filtered, probs):
    #     if conf['English'] < 0.8:
    #         print(conf)
    #         print(len(text))
    #         sample = text[:256].replace('\n', '')
    #         print(sample)
    #         print()

    filtered = [i for i, conf in zip(filtered, english_confs) if conf > lang_bar]
    if show_progress:
        print(f"Filtered: {len(filtered)} of {len(texts)}")
    return filtered


if __name__ == "__main__":
    texts = []
    with open("openwebtext2/downloaded/raw/test/part-00858-1ada68f1-85b8-473a-b6af-71a367a5ccbf.jsonl",
              encoding="utf-8") as f:
        for line in f:
            line_json = json.loads(line)
            if "content" in line_json:
                texts.append(line_json["content"])

    filtered = filter(texts, plot=True)
