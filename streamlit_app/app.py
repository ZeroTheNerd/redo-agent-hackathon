import os, sys, json, subprocess, pandas as pd, math, hashlib
import streamlit as st
from collections import Counter
from dotenv import load_dotenv

# ---------- Env / Paths ----------
load_dotenv()
RESULTS_PATH = "outputs/results.json"
PRODUCT_PATH = "product.json"
AGENTS_PATH = "outputs/agents.json"
SUGGEST_CACHE_DIR = "outputs/.suggest_cache"
os.makedirs(SUGGEST_CACHE_DIR, exist_ok=True)

# ---------- Page config ----------
st.set_page_config(page_title="AI Agent Product Forecaster", layout="wide")

# ---------- Helpers ----------
def product_hash(product: dict) -> str:
    return hashlib.sha256(json.dumps(product, sort_keys=True).encode()).hexdigest()

def write_product_json(name, price, features, positioning, comparables):
    product = {
        "name": (name or "").strip(),
        "price": float(price or 0.0),
        "features": [x.strip() for x in (features or "").split(",") if x.strip()],
        "positioning": (positioning or "").strip(),
        "comparables": [x.strip() for x in (comparables or "").split(",") if x.strip()],
    }
    with open(PRODUCT_PATH, "w") as f:
        json.dump(product, f, indent=2)
    return product

def run_pipeline():
    subprocess.run([sys.executable, "run.py"], check=False)

def load_json(path, default):
    if not os.path.exists(path): return default
    try:
        with open(path) as f: return json.load(f)
    except Exception: return default

def load_results():  return load_json(RESULTS_PATH, [])
def load_agents():   return load_json(AGENTS_PATH, [])

def summarize(results):
    if not results:
        return 0.0, 0.0, 0.0, [], []

    scores   = [r.get("pmf_score", 0.0) for r in results if r.get("pmf_score") is not None]
    confs    = [r.get("confidence", 0.0) for r in results]
    weights  = [1 / max(0.05, r.get("reliability", 0.3)) for r in results]

    def wavg(vals):
        num = sum(v*w for v, w in zip(vals, weights))
        den = sum(weights)
        return (num/den) if den else 0.0

    pmf = wavg(scores)
    cnf = wavg(confs)
    mean = pmf
    var  = (sum(w*((s-mean)**2) for s, w in zip(scores, weights)) / sum(weights)) if sum(weights) else 0.0
    unc  = math.sqrt(var)

    risks = Counter(x for r in results for x in r.get("risks", []) if x)
    asmp  = Counter(x for r in results for x in r.get("assumptions", []) if x)

    return pmf, cnf, unc, risks.most_common(5), asmp.most_common(5)

# ---- Suggestions cache helpers ----
def _suggest_cache_path(product: dict) -> str:
    return os.path.join(SUGGEST_CACHE_DIR, f"{product_hash(product)}.json")

def load_suggest_cache(product: dict):
    p = _suggest_cache_path(product)
    if os.path.exists(p):
        try: return json.load(open(p))
        except Exception: return None

def save_suggest_cache(product: dict, data: dict):
    with open(_suggest_cache_path(product), "w") as f:
        json.dump(data, f, indent=2)

def _normalize_suggestions(obj: dict) -> dict:
    def as_list(x):
        if x is None: return []
        if isinstance(x, list): return [str(i) for i in x if str(i).strip()]
        return [str(x)]
    return {
        "improvements": as_list(obj.get("improvements")),
        "alternative_products": as_list(obj.get("alternative_products")),
        "positioning_changes": as_list(obj.get("positioning_changes")),
    }

def suggest_product_changes(product, pmf, cnf, unc, top_risks, top_asmp):
    cached = load_suggest_cache(product)
    if cached:
        return _normalize_suggestions(cached), None

    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"improvements": [], "alternative_products": [], "positioning_changes": []}, "OPENAI_API_KEY is missing"
        client = OpenAI(api_key=api_key)

        prompt = f"""
You are analyzing how to improve product-market fit for the given personas.

[PRODUCT]
{json.dumps(product, indent=2)}

[FORECAST SUMMARY]
PMF Score: {pmf:.2f}
Confidence: {cnf:.2f}
Uncertainty: ±{unc:.2f}
Top Risks: {[r for r,_ in top_risks]}
Top Assumptions: {[a for a,_ in top_asmp]}

TASK:
Suggest:
1) Concrete product improvements (features, pricing, UX) that increase adoption for this demographic.
2) Alternative product variants / bundles that might fit better.
3) Positioning / marketing changes aligned to the personas' signals.

Return STRICT JSON only:
{{
  "improvements": ["..."],
  "alternative_products": ["..."],
  "positioning_changes": ["..."]
}}
"""
        resp = client.chat.completions.create(
            model=os.getenv("SUGGEST_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_object"}
        )
        raw = json.loads(resp.choices[0].message.content)
        norm = _normalize_suggestions(raw)
        save_suggest_cache(product, norm)
        return norm, None

    except Exception as e:
        return {"improvements": [], "alternative_products": [], "positioning_changes": []}, f"{type(e).__name__}: {e}"

# ---------- UI: Inputs ----------
with st.sidebar:
    st.header("New Product")
    name = st.text_input("Name", value="Smart Hydration Tracker")
    price = st.number_input("Price ($)", min_value=0.0, value=59.99, step=0.5)
    features = st.text_area("Features (comma-separated)", value="tracks water intake, glows as reminder, syncs with phone, eco-friendly material")
    positioning = st.text_area("Positioning", value="Designed for active, health-conscious individuals")
    comparables = st.text_input("Comparables (comma-separated)", value="Hidrate Spark, Larq Bottle")
    run_btn = st.button("🚀 Suggest / Forecast", use_container_width=True)

st.title("AI Crowd of Agents — Product Fit Forecast")

# ---------- Run + Display ----------
if run_btn:
    product = write_product_json(name, price, features, positioning, comparables)
    with st.spinner("Running agents (map) and aggregator (reduce)…"):
        run_pipeline()

# Always try to show latest results if present
results = load_results()
agents_cache = load_agents()  # used to show profile/evidence for a selected persona
agents_by_id = {a.get("agent_id"): a for a in agents_cache}

if not results:
    st.info("Click **Suggest / Forecast** to run the agents.")
else:
    pmf, cnf, unc, top_risks, top_asmp = summarize(results)

    # KPI cards
    c1, c2, c3 = st.columns(3)
    c1.metric("PMF Score", f"{pmf:.2f}")
    c2.metric("Confidence", f"{cnf:.2f}")
    c3.metric("Uncertainty (±)", f"{unc:.2f}")

    # -------- Persona Scores (search + selectable table) --------
    st.subheader("Persona Scores")

    # Build table data
    rows = [{
        "Agent": r.get("agent_id", "Unknown"),
        "PMF": round(float(r.get("pmf_score", 0.0)), 3) if r.get("pmf_score") is not None else None,
        "Confidence": round(float(r.get("confidence", 0.0)), 3) if r.get("confidence") is not None else None,
        "Reliability": r.get("reliability", None),
        "Rationale": r.get("rationale", ""),
        "Risks": " • ".join(r.get("risks", [])[:5]),
        "Assumptions": " • ".join(r.get("assumptions", [])[:5]),
    } for r in results]

    table_df = pd.DataFrame(rows).dropna(subset=["PMF"])

    # Quick search/filter
    q = st.text_input("Filter personas by name (case-insensitive):", "")
    if q:
        mask = table_df["Agent"].str.contains(q, case=False, na=False)
        table_df = table_df[mask]

    # Bar chart stays for overview
    if not table_df.empty:
        st.bar_chart(table_df.set_index("Agent")["PMF"])

    # Selectable table
    st.caption("Tip: select a row to open a detailed view.")
    edited = st.data_editor(
        table_df,
        hide_index=True,
        disabled=["PMF", "Confidence", "Reliability", "Rationale", "Risks", "Assumptions"],
        use_container_width=True,
        key="agent_table",
    )

    # Figure out selected row(s)
    sel = st.session_state.get("agent_table", {}).get("selection", {}).get("rows", [])
    if sel:
        idx = list(sel)[0]  # take the first selected row
        sel_agent = table_df.iloc[idx]["Agent"]

        # Open a modal with details
        with st.modal(f"Persona details — {sel_agent}"):
            # Agent result
            r = next((x for x in results if x.get("agent_id") == sel_agent), None)
            # Agent cached profile/evidence
            a = agents_by_id.get(sel_agent, {})

            c1, c2, c3 = st.columns(3)
            c1.metric("PMF", f"{r.get('pmf_score', 0):.2f}" if r else "—")
            c2.metric("Confidence", f"{r.get('confidence', 0):.2f}" if r else "—")
            c3.metric("Reliability", f"{r.get('reliability', 0.3):.2f}" if r else "—")

            st.markdown("#### Rationale")
            st.write(r.get("rationale", "(none)") if r else "(none)")

            rc, ac = st.columns(2)
            with rc:
                st.markdown("#### Risks")
                st.write(r.get("risks", []) if r else [])
            with ac:
                st.markdown("#### Assumptions")
                st.write(r.get("assumptions", []) if r else [])

            st.divider()
            st.markdown("### Profile & Evidence (from agents cache)")

            pc1, pc2, pc3 = st.columns(3)
            prof = a.get("profile", {})
            pc1.write({"Age": prof.get("age"), "Location": prof.get("location"), "Occupation": prof.get("occupation")})
            pc2.write({"Tech Savvy": prof.get("tech_savvy"), "Eco Conscious": prof.get("eco_conscious"), "Health Focused": prof.get("health_focused")})
            pc3.write({"Price Sensitive": prof.get("price_sensitive"), "Early Adopter": prof.get("early_adopter"), "Top Channel": prof.get("top_channel")})

            st.markdown("#### Outreach")
            st.write(a.get("outreach", {}))
            st.markdown("#### Store")
            st.write(a.get("store", {}))
            st.markdown("#### Orders")
            st.write(a.get("orders", {}))
            st.markdown("#### Derived Signals")
            st.write(a.get("derived", {}))

    # -------- Risks & assumptions (global) --------
    rcol, acol = st.columns(2)
    with rcol:
        st.subheader("Top Risks")
        st.write([k for k,_ in top_risks] or ["(none)"])
    with acol:
        st.subheader("Top Assumptions")
        st.write([k for k,_ in top_asmp] or ["(none)"])

    # -------- Suggestions section --------
    st.subheader("💡 Suggestions to Improve Product Fit")
    if os.path.exists(PRODUCT_PATH):
        product = json.load(open(PRODUCT_PATH))
    else:
        product = write_product_json(name, price, features, positioning, comparables)

    suggestions, err = suggest_product_changes(product, pmf, cnf, unc, top_risks, top_asmp)
    if err:
        st.warning(f"Suggestions could not be generated: {err}")

    st.write("- **Improvements:**", suggestions.get("improvements") or ["(no suggestions)"])
    st.write("- **Alternative products:**", suggestions.get("alternative_products") or ["(no suggestions)"])
    st.write("- **Positioning changes:**", suggestions.get("positioning_changes") or ["(no suggestions)"])

    # -------- MCP internals (debug) --------
    with st.expander("🔎 Per-agent JSON (debug)"):
        st.json(results)
