import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-20b"
CONSISTENCY_RUNS = 3
DELAY_SECONDS = 2.0

DATA_PATH = "data/test_prompts.csv"
OUTPUT_PATH = "outputs/raw_responses.csv"
REQUIRED_COLUMNS = {"prompt_id", "category", "prompt_text"}


def load_prompts(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"test_prompts.csv is missing columns: {missing}")
    return df


def call_groq(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def run_evaluation(df: pd.DataFrame) -> list[dict]:
    results = []
    total = len(df)

    for idx, row in df.iterrows():
        is_consistency = str(row["category"]).strip().upper() == "C"
        runs = CONSISTENCY_RUNS if is_consistency else 1
        label = f"[{row['prompt_id']}] ({row['category']})"

        for run_num in range(1, runs + 1):
            suffix = f" run {run_num}/{runs}" if runs > 1 else ""
            print(f"  {label}{suffix} ...", end=" ", flush=True)

            try:
                response_text = call_groq(str(row["prompt_text"]))
                print("OK")
            except Exception as exc:
                response_text = f"ERROR: {exc}"
                print(f"FAILED — {exc}")

            results.append({
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "prompt_text": row["prompt_text"],
                "response": response_text,
                "run_number": run_num,
            })

            time.sleep(DELAY_SECONDS)

        print(f"  Progress: {idx + 1}/{total} prompts done\n")

    return results


def save_results(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    print("=== LLM Evaluation Run ===\n")

    print(f"Loading prompts: {DATA_PATH}")
    df = load_prompts(DATA_PATH)
    print(f"  {len(df)} prompts loaded\n")

    print(f"Model: {MODEL_NAME}\n")

    results = run_evaluation(df)

    errors = sum(1 for r in results if str(r["response"]).startswith("ERROR:"))
    print(f"=== Done: {len(results)} responses recorded, {errors} error(s) ===")

    save_results(results, OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
