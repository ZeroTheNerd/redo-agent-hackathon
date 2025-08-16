import os
import json
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# Load OpenAI key from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError(" OPENAI_API_KEY not found in .env file.")

client = OpenAI(api_key=openai_key)

# Load persona data
df = pd.read_excel("data/personas.xlsx")
product = json.load(open("product.json"))
template = open("prompt_template.txt").read()

# Ensure outputs/ folder exists
os.makedirs("outputs", exist_ok=True)

def safe(val):
    return str(val) if pd.notna(val) else ""

def render_prompt(row):
    return template \
        .replace("{{name}}", safe(row.get("Name"))) \
        .replace("{{age}}", safe(row.get("Age"))) \
        .replace("{{location}}", safe(row.get("Location"))) \
        .replace("{{interests}}", safe(row.get("Interests"))) \
        .replace("{{channel}}", safe(row.get("Channel"))) \
        .replace("{{product.name}}", product["name"]) \
        .replace("{{product.price}}", str(product["price"])) \
        .replace("{{product.features}}", ", ".join(product["features"])) \
        .replace("{{product.positioning}}", product["positioning"]) \
        .replace("{{product.comparables}}", ", ".join(product["comparables"]))

def ask_agent(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"}  # force JSON response
    )
    return json.loads(response.choices[0].message.content)

results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring agents"):
    try:
        prompt = render_prompt(row)
        res = ask_agent(prompt)
        res["reliability"] = float(row.get("Reliability", 0.3))
        results.append(res)
    except Exception as e:
        print(f"[ Error for agent {row.get('Name')}]:", e)

# Save final result to disk
with open("outputs/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n Saved {len(results)} agent results to outputs/results.json")
