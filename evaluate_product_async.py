# evaluate_product_async.py
import os, json, asyncio, hashlib, random, math
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")
client = AsyncOpenAI(api_key=API_KEY)

MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")   # fast map model
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "16"))
MAX_RETRIES = 3
TIMEOUT_S = 60

OUT_DIR = Path("outputs")
AGENTS_PATH = OUT_DIR / "agents.json"
RESULTS_PATH = OUT_DIR / "results.json"
EVAL_CACHE_DIR = OUT_DIR / ".eval_cache"
EVAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -------- product payload & hash --------
product = json.load(open("product.json"))

def product_hash():
    return hashlib.sha256(json.dumps(product, sort_keys=True).encode()).hexdigest()

# -------- prompting --------
SYSTEM = (
    "You are scoring product-market fit for ONE persona using ONLY the given JSON.\n"
    "Return STRICT JSON with numeric fields (no strings for numbers).\n"
    "Schema:\n"
    "{\n"
    "  \"agent_id\": string,\n"
    "  \"pmf_score\": number (0..1),\n"
    "  \"confidence\": number (0..1),\n"
    "  \"rationale\": string,\n"
    "  \"risks\": string[],\n"
    "  \"assumptions\": string[]\n"
    "}\n"
    "Rules:\n"
    "- If signals are sparse/unknown, lower confidence.\n"
    "- Tie risks/assumptions to the provided agent signals only.\n"
)

def messages_for(agent):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps({
            "agent": {
                "agent_id": agent["agent_id"],
                "profile": agent["profile"],
                "orders": agent["orders"],
                "outreach": agent["outreach"],
                "store": agent["store"],
                "derived": agent["derived"],
            },
            "product": product
        }, ensure_ascii=False)}
    ]

# -------- caching per (agent, product) --------
def cache_path(agent_hash: str) -> Path:
    key = hashlib.sha256(f"{agent_hash}:{product_hash()}".encode()).hexdigest()
    return EVAL_CACHE_DIR / f"{key}.json"

def try_load_eval(agent_hash: str):
    p = cache_path(agent_hash)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

def save_eval(agent_hash: str, obj: dict):
    cache_path(agent_hash).write_text(json.dumps(obj))

# -------- normalization & safety --------
def to_float01(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, v))

def normalize_record(raw: dict, agent: dict) -> dict:
    out = {}
    out["agent_id"]   = raw.get("agent_id") or agent.get("agent_id") or "unknown"
    out["pmf_score"]  = to_float01(raw.get("pmf_score"))
    out["confidence"] = to_float01(raw.get("confidence"))
    out["rationale"]  = raw.get("rationale") or ""
    out["risks"]      = raw.get("risks") or []
    out["assumptions"]= raw.get("assumptions") or []
    out["reliability"]= agent.get("reliability", 0.3)

    # fallback defaults so reducer never breaks
    if out["pmf_score"] is None:
        out["pmf_score"] = 0.0
        if not out["rationale"]:
            out["rationale"] = "missing/invalid pmf_score; defaulted to 0.0"
    if out["confidence"] is None:
        out["confidence"] = 0.1
    # ensure list types
    if not isinstance(out["risks"], list): out["risks"] = [str(out["risks"])]
    if not isinstance(out["assumptions"], list): out["assumptions"] = [str(out["assumptions"])]
    return out

# -------- one agent eval --------
async def eval_one(agent, sem):
    # per-product cache first
    cached = try_load_eval(agent["agent_hash"])
    if cached:
        return normalize_record(cached, agent), "cached"

    status = "ok"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages_for(agent),
                    temperature=0.15,
                    response_format={"type": "json_object"}
                )
            raw = json.loads(resp.choices[0].message.content)
            rec = normalize_record(raw, agent)
            save_eval(agent["agent_hash"], rec)
            return rec, status
        except Exception as e:
            status = f"retry{attempt}"
            if attempt == MAX_RETRIES:
                # hard-fail but still return a valid record
                rec = {
                    "agent_id": agent["agent_id"],
                    "pmf_score": 0.0,
                    "confidence": 0.1,
                    "rationale": f"error: {e}",
                    "risks": [],
                    "assumptions": [],
                    "reliability": agent.get("reliability", 0.3),
                }
                save_eval(agent["agent_hash"], rec)  # cache the fallback too
                return rec, "failed"
            await asyncio.sleep((2 ** (attempt - 1)) + random.random())

# -------- main --------
async def main():
    if not AGENTS_PATH.exists():
        raise RuntimeError("outputs/agents.json not found. Run build_agents.py first.")

    agents = json.load(open(AGENTS_PATH))
    if not agents:
        RESULTS_PATH.write_text("[]")
        print("No agents to evaluate. Wrote empty results.")
        return

    sem = asyncio.Semaphore(max(1, min(MAX_CONCURRENCY, len(agents))))
    tasks = [asyncio.create_task(eval_one(a, sem)) for a in agents]
    results_with_status = await asyncio.gather(*tasks)

    results = []
    stats = {"ok": 0, "cached": 0, "failed": 0, "retry1": 0, "retry2": 0, "retry3": 0}
    for rec, status in results_with_status:
        results.append(rec)
        stats[status] = stats.get(status, 0) + 1

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    total = len(results)
    print(f"Saved {total} results → {RESULTS_PATH}")
    print("Eval stats:", {k: v for k, v in stats.items() if v})

if __name__ == "__main__":
    asyncio.run(main())
