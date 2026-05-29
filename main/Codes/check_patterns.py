"""Quick check: which participants show the expected EEG pattern?"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "results_analysis", "subband_analysis.csv"))
binary = df[df.Complexity.isin(["Simple", "Complex"])]
participants = sorted(binary.Participant.unique())

print("=" * 90)
print("PER-PARTICIPANT PATTERN CHECK: Does Complex show expected EEG pattern?")
print("Expected: Theta DOWN, Alpha DOWN, Beta UP (vs Simple)")
print("=" * 90)

results = []
for p in participants:
    pdf = binary[binary.Participant == p]
    s = pdf[pdf.Complexity == "Simple"]
    c = pdf[pdf.Complexity == "Complex"]
    if len(s) == 0 or len(c) == 0:
        continue

    theta_s, theta_c = s["Theta"].mean(), c["Theta"].mean()
    la_s, la_c = s["Low Alpha"].mean(), c["Low Alpha"].mean()
    ha_s, ha_c = s["High Alpha"].mean(), c["High Alpha"].mean()
    lb_s, lb_c = s["Low Beta"].mean(), c["Low Beta"].mean()
    hb_s, hb_c = s["High Beta"].mean(), c["High Beta"].mean()

    theta_down = theta_c < theta_s
    alpha_down = (la_c + ha_c) < (la_s + ha_s)
    beta_up = (lb_c + hb_c) > (lb_s + hb_s)
    full_pattern = theta_down and alpha_down and beta_up

    theta_pct = ((theta_c - theta_s) / (theta_s + 1e-10)) * 100
    alpha_pct = (((la_c + ha_c) - (la_s + ha_s)) / ((la_s + ha_s) + 1e-10)) * 100
    beta_pct = (((lb_c + hb_c) - (lb_s + hb_s)) / ((lb_s + hb_s) + 1e-10)) * 100

    results.append({
        "Participant": p, "Theta_down": theta_down, "Alpha_down": alpha_down,
        "Beta_up": beta_up, "Full_pattern": full_pattern,
        "Theta_pct": theta_pct, "Alpha_pct": alpha_pct, "Beta_pct": beta_pct,
    })

rdf = pd.DataFrame(results)
n = len(rdf)
n_theta = rdf.Theta_down.sum()
n_alpha = rdf.Alpha_down.sum()
n_beta = rdf.Beta_up.sum()
n_full = rdf.Full_pattern.sum()

print(f"\nTotal participants with both classes: {n}")
print(f"")
print(f"  Theta suppression (Complex < Simple):  {n_theta}/{n} ({n_theta/n*100:.0f}%)")
print(f"  Alpha suppression (Complex < Simple):  {n_alpha}/{n} ({n_alpha/n*100:.0f}%)")
print(f"  Beta enhancement  (Complex > Simple):  {n_beta}/{n} ({n_beta/n*100:.0f}%)")
print(f"  FULL expected pattern (all 3 correct): {n_full}/{n} ({n_full/n*100:.0f}%)")

print(f"\n--- FULL PATTERN (theta down + alpha down + beta up) --- [{n_full} participants]")
full = rdf[rdf.Full_pattern].sort_values("Theta_pct")
for _, r in full.iterrows():
    print(f"  {r.Participant:<14s}  Theta:{r.Theta_pct:+6.0f}%  Alpha:{r.Alpha_pct:+6.0f}%  Beta:{r.Beta_pct:+6.0f}%")

print(f"\n--- PARTIAL PATTERN (1-2 of 3 correct) ---")
score = rdf[["Theta_down", "Alpha_down", "Beta_up"]].sum(axis=1)
partial = rdf[(~rdf.Full_pattern) & (score > 0)].sort_values("Theta_pct")
for _, r in partial.iterrows():
    t = "DOWN" if r.Theta_down else " UP "
    a = "DOWN" if r.Alpha_down else " UP "
    b = " UP " if r.Beta_up else "DOWN"
    print(f"  {r.Participant:<14s}  theta={t} alpha={a} beta={b}  |  Theta:{r.Theta_pct:+6.0f}%  Alpha:{r.Alpha_pct:+6.0f}%  Beta:{r.Beta_pct:+6.0f}%")

print(f"\n--- OPPOSITE PATTERN (0 of 3 correct: theta UP, alpha UP, beta DOWN) ---")
opposite = rdf[score == 0]
for _, r in opposite.iterrows():
    print(f"  {r.Participant:<14s}  Theta:{r.Theta_pct:+6.0f}%  Alpha:{r.Alpha_pct:+6.0f}%  Beta:{r.Beta_pct:+6.0f}%")
