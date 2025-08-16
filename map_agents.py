import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load env and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load event data only (no personas.xlsx)
orders = pd.read_excel("data/Order Events.xlsx")
outreach = pd.read_excel("data/Outreach Events.xlsx")
store = pd.read_excel("data/Store Events.xlsx")

# Merge all event data on 'Name'
df = orders.merge(outreach, on="Name", how="outer")
df = df.merge(store, on="Name", how="outer")

# Load product + prompt template
product = json.load(open("product.json"))
template = open("prompt_template.txt").read()

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

def safe(val):
    return str(val) if pd.notna(val) else "N/A"

def render_prompt(row):
    # Format evidence block
    evidence = f"""
[ORDER EVENTS]
Units Sold: {safe(row.get('Units Sold'))}
Revenue: {safe(row.get('Revenue'))}
AOV: {safe(row.get('AOV'))}

[OUTREACH EVENTS]
Impressions: {safe(row.get('Impressions'))}
CTR: {safe(row.get('CTR'))}
CVR: {safe(row.get('CVR'))}

[STORE EVENTS]
Page Views: {safe(row.get('Page Views'))}
Add-to-Cart Rate: {safe(row.get('Add to Cart Rate'))}
Checkout Rate: {safe(row.get('Checkout Rate'))}
"""

    prompt = template \
        .replace("{{name}}", safe(row["Name"])) \
        .replace("{{age}}", safe(row.get("Age", "N/A"))) \
        .replace("{{location}}", safe(row.get("Location", "N/A"))) \
        .replace("{{interests}}", safe(row.get("Interests", "N/A"))) \
        .replace("{{channel}}", safe(row.get("Channel", "N/A"))) \
        .replace("{{product.name}}", product["name"]) \
        .replace("{{product.price}}", str(product["price"])) \
        .replace("{{product.features}}", ", ".join(product["features"])) \
        .replace("{{product.positioning}}", product["positioning"]) \
        .replace("{{product.comparables}}", ", ".join(product["comparables"])) \
        .replace("{{evidence_block}}", evidence)

    return prompt

def ask_agent(prompt):
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    return json.loads(res.choices[0].message.content)

results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring agents"):
    try:
        prompt = render_prompt(row)
        res = ask_agent(prompt)
        res["reliability"] = float(row.get("Reliability", 0.3))
        results.append(res)
    except Exception as e:
        print(f"[ Error for agent {row.get('Name')}]:", e)

with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Saved {len(results)} agent results to outputs/results.json")
