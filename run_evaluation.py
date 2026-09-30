import os
from dotenv import load_dotenv
load_dotenv()
import time
import pandas as pd
from groq import Groq

# ── Settings ────────────────────────────────────────────────────────────────
MODEL_NAME = "openai/gpt-oss-20b"
CONSISTENCY_RUNS = 3   # how many times to repeat consistency-category prompts
DELAY_SECONDS = 2.0    # pause between API calls to stay within rate limits

DATA_PATH = "data/test_prompts.csv"
OUTPUT_PATH = "outputs/raw_responses.csv"

REQUIRED_COLUMNS = {"prompt_id", "category", "prompt_text"}
# ────────────────────────────────────────────────────────────────────────────


def load_prompts(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"test_prompts.csv is missing columns: {missing}")
    return df


def call_groq(client: Groq, prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as exc:
        if "429" in str(exc):
            print("rate-limited (429), waiting 60 s ...", end=" ", flush=True)
            time.sleep(60)
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        raise


def run_evaluation(df: pd.DataFrame, client: Groq) -> list[dict]:
    results = []
    total_prompts = len(df)

    for idx, row in df.iterrows():
        runs = CONSISTENCY_RUNS if str(row["category"]).lower() == "consistency" else 1
        label = f"[{row['prompt_id']}] ({row['category']})"

        for run_num in range(1, runs + 1):
            run_label = f"run {run_num}/{runs}" if runs > 1 else ""
            print(f"  {label} {run_label}".rstrip(), end=" ... ", flush=True)

            try:
                response_text = call_groq(client, row["prompt_text"])
                status = "OK"
            except Exception as exc:
                response_text = f"ERROR: {exc}"
                status = f"FAILED — {exc}"

            print(status)

            results.append({
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "prompt_text": row["prompt_text"],
                "response": response_text,
                "run_number": run_num,
            })

            time.sleep(DELAY_SECONDS)

        progress = idx + 1
        print(f"  Progress: {progress}/{total_prompts} prompts done\n")

    return results


def save_results(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    print("=== LLM Evaluation Run ===\n")

    print(f"Loading prompts from: {DATA_PATH}")
    df = load_prompts(DATA_PATH)
    print(f"  {len(df)} prompts loaded\n")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print(f"Model: {MODEL_NAME}\n")

    results = run_evaluation(df, client)

    errors = sum(1 for r in results if r["response"].startswith("ERROR:"))
    print(f"=== Run complete: {len(results)} responses, {errors} error(s) ===")

    save_results(results, OUTPUT_PATH)
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
