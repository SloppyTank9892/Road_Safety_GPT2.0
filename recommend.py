#!/usr/bin/env python3
import argparse
import pandas as pd
from rs_gpt import RSInterventionGPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models", help="Model directory")
    ap.add_argument("--query", required=True, help="Free-text issue description")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--category", help="Optional category filter")
    ap.add_argument("--type", dest="type_filter", help="Optional type filter")
    ap.add_argument("--save", help="Optional path to save results CSV")
    args = ap.parse_args()

    model = RSInterventionGPT.load(args.model)
    res = model.recommend(args.query, top_k=args.topk, category=args.category, type_filter=args.type_filter)
    if res.empty:
        print("No matches found.")
        return

    # Display nicely with percentages
    pd.set_option('display.max_colwidth', None)
    formatted = res.copy()
    if 'score' in formatted.columns:
        formatted['score'] = formatted['score'].apply(lambda x: f"{x*100:.1f}%")

    print("\nRecommended Interventions:\n")
    for _, row in formatted.iterrows():
        print("-" * 80)
        print(f"Rank {row['rank']} (Score: {row['score']})")
        print(f"Problem: {row['problem']}")
        print(f"Category: {row['category']} | Type: {row['type']}")
        print("\nExplanation:\n")
        print(row['explanation'])
        print(f"\nReference Code: {row['reference_code']}")
        print(f"Reference Clause: {row['reference_clause']}")
    print("-" * 80)

    if args.save:
        # save original numeric scores
        res.to_csv(args.save, index=False)
        print(f"\nSaved results to: {args.save}")


if __name__ == "__main__":
    main()
