"""
Usability Study Analysis — Campus Food Security App (HCI project)
=================================================================

Analyzes task-based usability-survey data from 10 participants and produces the
quantitative evidence used to prioritize design changes:

    1. Cleans messy free-text satisfaction ratings into a consistent 0-10 scale.
    2. Computes the proportion of participants who found each key step clear/easy.
    3. Codes open-ended feedback into friction themes and counts how many
       participants mentioned each one.

Two figures are written to ./figures and summary stats are printed to stdout.

Usage
-----
    pip install -r requirements.txt
    python usability_analysis.py

Expects
-------
    data/usability_all_data.csv  with (at least) the columns:
        participant_id, question_text, answer_text
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "data/usability_all_data.csv"
FIG_DIR = "figures"
N_PARTICIPANTS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def classify_answer(ans):
    """Classify a free-text answer as 'yes' / 'no' / 'other' / 'no_response'."""
    if not isinstance(ans, str) or ans.strip() == "" or str(ans).lower().startswith("nan"):
        return "no_response"
    s = ans.strip().lower()
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no") or s.startswith("not"):
        return "no"
    return "other"


def parse_rating_to_10(text):
    """Turn messy answers like '8 out of 10', '4/5', or '4.5' into a 0-10 score.

    Rules (applied in order):
      - "<x> out of <y>"  -> x * 10 / y
      - "<x>/<y>"         -> x * 10 / y
      - a bare number     -> if <= 5 assume it was out of 5 (so * 2), else as-is
    Returns np.nan when no number can be found.
    """
    if not isinstance(text, str):
        return np.nan
    s = text.strip().lower()
    if not s or s == "nan":
        return np.nan

    m = re.search(r"(\d+(?:\.\d+)?)\s*out of\s*(\d+)", s)
    if m:
        val, denom = float(m.group(1)), float(m.group(2))
        return val * 10.0 / denom if denom != 0 else np.nan

    m = re.search(r"(\d+(?:\.\d+)?)/(\d+)", s)
    if m:
        val, denom = float(m.group(1)), float(m.group(2))
        return val * 10.0 / denom if denom != 0 else np.nan

    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        val = float(m.group(1))
        return val * 2.0 if val <= 5 else val

    return np.nan


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def parse_ratings(df):
    """Parse the overall-experience rating question into 0-10 scores."""
    rating_q = "How would you rate the systems based on this experience?"
    rating_df = df[df["question_text"] == rating_q].copy()
    rating_df["rating_10"] = rating_df["answer_text"].apply(parse_rating_to_10)

    print("=== Overall experience ratings (parsed to 0-10) ===")
    print(rating_df[["participant_id", "answer_text", "rating_10"]].to_string(index=False))
    mean_rating = rating_df["rating_10"].mean()
    print(f"\nMean rating (where a number was given): {mean_rating:.1f} / 10\n")
    return rating_df


def clarity_by_step(df):
    """Proportion of participants giving an explicit 'Yes' at each key step."""
    # Map a short, readable label to the (long) survey question via a prefix.
    prefixes = {
        "Understood overall flow":            "When you opened and used the app for the fi",
        "Map made locker selection easy":     "Did the map make it easy for you to select a locker",
        "Sections made finding items easy":   "Did the sections in the app make it easy for you to find",
        "Clear when items added to bin":      "Was it clear to you when your selections were successfully",
        "Clear when checkout completed":      "Was it clear when you had successfully compl",
        "Confident order details were correct": "After completing the checkout, did you feel confident",
    }

    labels, yes_props = [], []
    for label, prefix in prefixes.items():
        matches = [q for q in df["question_text"].unique()
                   if isinstance(q, str) and q.startswith(prefix)]
        if not matches:
            print(f"[warn] no question matched prefix for: {label}")
            continue
        subset = df[df["question_text"] == matches[0]]
        cats = subset["answer_text"].apply(classify_answer)
        total = len(subset)
        labels.append(label)
        yes_props.append((cats == "yes").sum() / total if total else np.nan)

    print("=== Perceived ease & clarity at key steps (explicit 'Yes') ===")
    for label, prop in zip(labels, yes_props):
        print(f"{label}: {prop * 100:.0f}% (out of {N_PARTICIPANTS})")
    print()

    plt.figure(figsize=(8, 4))
    x = np.arange(len(labels))
    plt.bar(x, yes_props)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Proportion answering 'Yes'")
    plt.title("Perceived ease and clarity at key steps")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "clarity_by_step.png"), dpi=150)
    plt.close()
    return dict(zip(labels, yes_props))


def friction_analysis(df):
    """Code open-ended feedback into friction themes and count mentions."""
    friction_q_texts = [
        "Were there any difficulties using the app, such as selecting items, "
        "scheduling pick-up, or navigating the map, at any point?",
        "Is there any part that makes you confused or that you just dislike about it?",
        "Is there any part that makes you confused or that you just dislike about it?\xa0Any Suggestions?",
        "Anything else that stood out to you? Any suggestions?",
    ]
    friction_df = df[df["question_text"].isin(friction_q_texts)].copy()

    # one combined free-text blob per participant
    per_participant = (
        friction_df.groupby("participant_id")["answer_text"]
        .apply(lambda xs: " ".join(str(x) for x in xs if isinstance(x, str)))
        .reset_index()
    )

    feature_keywords = {
        "Map / locker / location":        ["map", "locker", "location", "pin"],
        "Time / scheduling":              ["time", "schedule", "scheduling", "pickup time", "pick-up time"],
        "Verification / food bank survey": ["survey", "food bank", "verify", "verification"],
        "Item quantity / add-to-cart":    ["add to cart", "cart", "bin", "quantity", "quantitty", "+", " - "],
        "Sections / categories / labels": ["category", "categories", "section", "sections",
                                           "label", "placeholder", "icons", "wordy"],
        "Confirmation / QR":              ["confirm", "confirmation", "qr"],
    }

    counts = {feat: 0 for feat in feature_keywords}
    for _, row in per_participant.iterrows():
        text = str(row["answer_text"]).lower()
        for feat, kws in feature_keywords.items():
            if any(kw in text for kw in kws):
                counts[feat] += 1

    n = len(per_participant)
    print("=== Friction areas (participants mentioning each) ===")
    for feat, c in counts.items():
        print(f"{feat}: {c} / {n}")
    print()

    feats = list(counts.keys())
    vals = [counts[f] for f in feats]
    plt.figure(figsize=(8, 4))
    x = np.arange(len(feats))
    plt.bar(x, vals)
    plt.xticks(x, feats, rotation=45, ha="right")
    plt.ylabel("Participants mentioning this area")
    plt.title("Features most often mentioned as confusing / needing improvement")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "friction_areas.png"), dpi=150)
    plt.close()
    return counts


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    parse_ratings(df)
    clarity_by_step(df)
    friction_analysis(df)
    print(f"Figures saved to ./{FIG_DIR}/")


if __name__ == "__main__":
    main()
