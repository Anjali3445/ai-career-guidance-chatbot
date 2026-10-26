"""Check that your Gemini API key works and see which models you can use.  Run:  python test_api.py"""
import os
import time

from dotenv import load_dotenv

import career_engine as ce

load_dotenv()
key = os.getenv("GEMINI_API_KEY", "")
if not key:
    raise SystemExit("GEMINI_API_KEY not found. Check your .env file.")

client = ce.get_client(key)

print("Flash models available for your key:")
found = ce._discover_models(client)
print("  " + (", ".join(found) if found else "(could not list models)"))

to_test = list(dict.fromkeys(ce.MODELS + found))
print("\nTesting models...")
working = []
for model in to_test:
    t = time.time()
    try:
        r = client.models.generate_content(
            model=model, contents="Reply with the single word: OK",
            config=ce._config(model, max_output_tokens=200))
        print(f"[WORKS]  {model}: {(r.text or '').strip()!r} in {time.time() - t:.1f}s")
        working.append(model)
    except Exception as e:
        print(f"[FAILED] {model} after {time.time() - t:.1f}s -> {str(e)[:150]}")

if working:
    print(f"\nGood news. To force a model, add this line to your .env file:\nGEMINI_MODEL={working[0]}")
else:
    print("\nNo model worked. Check your API key, internet, and free quota (try again after a few minutes).")