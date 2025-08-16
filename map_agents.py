import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load env and API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load all sheets
orders = pd.read_excel("data/Order Events.xlsx")
outreach = pd.read_excel("data/Outreach Events.xlsx")
store = pd.read_excel("data/Store Events.xlsx")
demographics = pd.read_excel("data/Demographics.xlsx")
psychographics = pd.read_excel("data/Psychographics.xlsx")
behavior = pd.read_excel("data/Behavioral Tags.xlsx")

# Merge all on Name
df = orders.merge(outreach, on="Name", how="outer")
df = df.merge(store, on="Name", how="outer")
df = df.merge(demographics, on="Name", how="left")
df = df.merge(psychographics, on="Name", how="left")
df = df.merge(behavior, on="Name", how="left")

# Load product + template
product = json.load(open("product.json"))
template = open("prompt_template.txt").read()

os.makedirs("outputs", exist_ok=True)

def safe(val):
    return str(val) if pd.notna(val) else "N/A"

def render_prompt(row):
    # Profile signals
    profile = f"""
[PROFILE SIGNALS]
Age: {safe(row.get("Age"))}
Location: {safe(row.get("Location"))}
Occupation: {safe(row.get("Occupation"))}
Tech Savvy: {safe(row.get("Tech Savvy"))}
Eco Conscious: {safe(row.get("Eco Conscious"))}
Health Focused: {safe(row.get("Health Focused"))}
Price Sensitive: {safe(row.get("Price Sensitive"))}
Early Adopter: {safe(row.get("Early Adopter"))}
Income Bracket: {safe(row.get("Income Bracket"))}
Top Channel: {safe(row.get("Top Channel"))}
"""

    evidence = f"""
[ORDER EVENTS]
Requested Refund: {safe(row.get("Requested Refund"))}
Order Delivered: {safe(row.get("Order Delivered"))}
Viewed Order History: {safe(row.get("Viewed Order History"))}
Placed Order: {safe(row.get("Placed Order"))}
Returned Item: {safe(row.get("Returned Item"))}

[OUTREACH EVENTS]
Responded to SMS: {safe(row.get("Responded to SMS"))}
Ignored Ad: {safe(row.get("Ignored Ad"))}
Clicked SMS Link: {safe(row.get("Clicked SMS Link"))}
Engaged with Push Notification: {safe(row.get("Engaged with Push Notification"))}
Clicked Email: {safe(row.get("Clicked Email"))}

[STORE EVENTS]
Applied Discount Code: {safe(row.get("Applied Discount Code"))}
Filtered by Price: {safe(row.get("Filtered by Price"))}
Abandoned Cart: {safe(row.get("Abandoned Cart"))}
Browsed New Arrivals: {safe(row.get("Browsed New Arrivals"))}
Searched Store: {safe(row.get("Searched Store"))}
Read Product Reviews: {safe(row.get("Read Product Reviews"))}
"""

    return template \
        .replace("{{name}}", safe(row["Name"])) \
        .replace("{{product.name}}", product["name"]) \
        .replace("{{product.price}}", str(product["price"])) \
        .replace("{{product.features}}", ", ".join(product["features"])) \
        .replace("{{product.positioning}}", product["positioning"]) \
        .replace("{{product.comparables}}", ", ".join(product["comparables"])) \
        .replace("{{profile_block}}", profile) \
        .replace("{{evidence_block}}", evidence)

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
