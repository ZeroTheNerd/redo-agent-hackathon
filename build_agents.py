# build_agents.py
import os, json, hashlib, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime

pd.options.mode.copy_on_write = True

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
AGENTS_PATH = OUT_DIR / "agents.json"
META_PATH = OUT_DIR / "agents.meta.json"

# ---------------- Utilities ----------------
def read_xlsx(name):
    p = DATA_DIR / name
    return pd.read_excel(p) if p.exists() else pd.DataFrame()

def flag(v):
    if pd.isna(v): return None
    s = str(v).strip().lower()
    if s in ("yes", "y", "true", "1"):  return True
    if s in ("no", "n", "false", "0"):  return False
    return None

def clamp01(x, default=0.3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))

def file_content_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def dataset_hash():
    # Hash the actual contents of every .xlsx to detect true changes
    h = hashlib.sha256()
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.lower().endswith(".xlsx"): 
            continue
        p = DATA_DIR / fname
        h.update(fname.encode())
        h.update(file_content_hash(p).encode())
    return h.hexdigest()

def first_non_null(series):
    for v in series:
        if pd.notna(v):
            return v
    return np.nan

# Canonicalize column names (lightweight mapping for common variations)
CANON_MAP = {
    # join key
    "name": "Name",
    # demographics / psychographics
    "age": "Age",
    "location": "Location",
    "occupation": "Occupation",
    "tech savvy": "Tech Savvy",
    "eco conscious": "Eco Conscious",
    "health focused": "Health Focused",
    "price sensitive": "Price Sensitive",
    "early adopter": "Early Adopter",
    "income bracket": "Income Bracket",
    "top channel": "Top Channel",
    "reliability": "Reliability",
    # outreach
    "responded to sms": "Responded to SMS",
    "ignored ad": "Ignored Ad",
    "clicked sms link": "Clicked SMS Link",
    "engaged with push notification": "Engaged with Push Notification",
    "clicked email": "Clicked Email",
    # store
    "applied discount code": "Applied Discount Code",
    "filtered by price": "Filtered by Price",
    "abandoned cart": "Abandoned Cart",
    "browsed new arrivals": "Browsed New Arrivals",
    "searched store": "Searched Store",
    "read product reviews": "Read Product Reviews",
    # orders
    "requested refund": "Requested Refund",
    "order delivered": "Order Delivered",
    "viewed order history": "Viewed Order History",
    "placed order": "Placed Order",
    "returned item": "Returned Item",
}

def canonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    new_cols = {}
    for c in df.columns:
        key = str(c).strip().lower()
        new_cols[c] = CANON_MAP.get(key, c)  # map if known, else keep
    return df.rename(columns=new_cols)

# ---------------- Build ----------------
def build_agents():
    # Load & canonize
    orders = canonize_columns(read_xlsx("Order Events.xlsx"))
    outreach = canonize_columns(read_xlsx("Outreach Events.xlsx"))
    store = canonize_columns(read_xlsx("Store Events.xlsx"))
    demo = canonize_columns(read_xlsx("Demographics.xlsx"))
    psych = canonize_columns(read_xlsx("Psychographics.xlsx"))
    beh = canonize_columns(read_xlsx("Behavioral Tags.xlsx"))

    # Merge all on Name (outer to keep anyone who appears somewhere)
    df = orders.merge(outreach, on="Name", how="outer") \
               .merge(store, on="Name", how="outer") \
               .merge(demo, on="Name", how="left") \
               .merge(psych, on="Name", how="left") \
               .merge(beh, on="Name", how="left")

    # Trim & dedupe Names
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).str.strip()
        df = df[df["Name"].str.len() > 0]
    else:
        raise ValueError("No 'Name' column found across inputs. Ensure each sheet includes a Name.")

    # If multiple rows per Name from merges, reduce to first non-null per column
    df = df.groupby("Name", as_index=False).agg(first_non_null)

    # Build agents
    agents = []
    skipped = 0
    for _, r in df.iterrows():
        name = str(r.get("Name") or "").strip()
        if not name:
            skipped += 1
            continue

        # profile
        profile = {
            "age": r.get("Age"),
            "location": r.get("Location"),
            "occupation": r.get("Occupation"),
            "tech_savvy": flag(r.get("Tech Savvy")),
            "eco_conscious": flag(r.get("Eco Conscious")),
            "health_focused": flag(r.get("Health Focused")),
            "price_sensitive": flag(r.get("Price Sensitive")),
            "early_adopter": flag(r.get("Early Adopter")),
            "income_bracket": r.get("Income Bracket"),
            "top_channel": r.get("Top Channel"),
        }

        # events (booleans)
        orders_b = {
            "requested_refund": flag(r.get("Requested Refund")),
            "order_delivered": flag(r.get("Order Delivered")),
            "viewed_order_history": flag(r.get("Viewed Order History")),
            "placed_order": flag(r.get("Placed Order")),
            "returned_item": flag(r.get("Returned Item")),
        }
        outreach_b = {
            "responded_sms": flag(r.get("Responded to SMS")),
            "ignored_ad": flag(r.get("Ignored Ad")),
            "clicked_sms": flag(r.get("Clicked SMS Link")),
            "push_engaged": flag(r.get("Engaged with Push Notification")),
            "clicked_email": flag(r.get("Clicked Email")),
        }
        store_b = {
            "applied_code": flag(r.get("Applied Discount Code")),
            "filtered_price": flag(r.get("Filtered by Price")),
            "abandoned_cart": flag(r.get("Abandoned Cart")),
            "browsed_new": flag(r.get("Browsed New Arrivals")),
            "searched_store": flag(r.get("Searched Store")),
            "read_reviews": flag(r.get("Read Product Reviews")),
        }

        # derived helpers (robust to None)
        price_signal = any(x is True for x in (store_b["applied_code"], store_b["filtered_price"]))
        engagement_signal = any(x is True for x in (
            outreach_b["responded_sms"], outreach_b["clicked_sms"],
            outreach_b["clicked_email"], outreach_b["push_engaged"]
        ))
        returns_signal = any(x is True for x in (orders_b["requested_refund"], orders_b["returned_item"]))

        derived = {
            "price_signal": price_signal,
            "engagement_signal": engagement_signal,
            "returns_signal": returns_signal,
        }

        reliability = clamp01(r.get("Reliability", 0.3), default=0.3)

        agent_payload = {
            "agent_id": name,
            "profile": profile,
            "orders": orders_b,
            "outreach": outreach_b,
            "store": store_b,
            "derived": derived,
            "reliability": reliability,
        }

        ah = hashlib.sha256(json.dumps(agent_payload, sort_keys=True, default=str).encode()).hexdigest()
        agent_payload["agent_hash"] = ah
        agents.append(agent_payload)

    # Write outputs
    AGENTS_PATH.write_text(json.dumps(agents, indent=2))
    META_PATH.write_text(json.dumps({
        "dataset_hash": dataset_hash(),
        "count": len(agents),
        "skipped": skipped,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "schema_version": 1
    }, indent=2))

    # Build report
    print(f"Built {len(agents)} agents → {AGENTS_PATH}")
    if skipped:
        print(f"Skipped {skipped} rows without a valid Name.")
    # quick null-rate peek for key signals
    key_cols = ["Tech Savvy","Eco Conscious","Health Focused","Price Sensitive","Early Adopter",
                "Responded to SMS","Clicked SMS Link","Clicked Email",
                "Applied Discount Code","Filtered by Price","Abandoned Cart",
                "Requested Refund","Returned Item"]
    present = [c for c in key_cols if c in df.columns]
    if present:
        null_rates = {c: float(pd.isna(df[c]).mean()) for c in present}
        print("Null rates (sample):", {k: round(v, 2) for k, v in null_rates.items()})

if __name__ == "__main__":
    new_hash = dataset_hash()
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
        except Exception:
            meta = {}
        if meta.get("dataset_hash") == new_hash and AGENTS_PATH.exists():
            print("No data change detected — using cached agents.")
        else:
            build_agents()
    else:
        build_agents()
