import json, math
from collections import Counter

data = json.load(open("outputs/results.json"))

def weighted_avg(vals, weights):
    num = sum(v * w for v, w in zip(vals, weights))
    den = sum(weights)
    return num / den if den > 0 else None

def weighted_std(vals, weights, mean):
    var = sum(w * (v - mean)**2 for v, w in zip(vals, weights)) / sum(weights)
    return math.sqrt(var)

scores = [d["pmf_score"] for d in data]
weights = [1 / max(0.05, d["reliability"]) for d in data]
confidences = [d["confidence"] for d in data]

pmf = weighted_avg(scores, weights)
conf = weighted_avg(confidences, weights)
uncertainty = weighted_std(scores, weights, pmf)

risks = Counter(r for d in data for r in d["risks"])
assumptions = Counter(a for d in data for a in d["assumptions"])

print("\n Product Forecast Summary")
print(f"PMF Score: {pmf:.2f}")
print(f"Confidence: {conf:.2f}")
print(f"Uncertainty: ±{uncertainty:.2f}")
print("\nTop Risks:", [r for r,_ in risks.most_common(3)])
print("Top Assumptions:", [a for a,_ in assumptions.most_common(3)])
