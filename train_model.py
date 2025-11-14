#!/usr/bin/env python3
import argparse
from pathlib import Path
from rs_gpt import RSInterventionGPT, RSConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="GPT_Input_DB.xlsx", help="Path to Excel/CSV database")
    ap.add_argument("--out", default="models", help="Output folder for artifacts")
    ap.add_argument("--no-embeddings", action="store_true", help="Disable sentence-transformer embeddings even if installed")
    args = ap.parse_args()

    cfg = RSConfig(enable_embeddings=not args.no_embeddings) if hasattr(RSConfig, "enable_embeddings") else RSConfig()
    model = RSInterventionGPT(cfg)
    model.fit(args.db)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved trained model to: {args.out}")

if __name__ == "__main__":
    main()
