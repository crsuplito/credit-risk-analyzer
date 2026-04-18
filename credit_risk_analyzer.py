"""
Credit Risk Analyzer
====================
A beginner-friendly Python tool for analyzing credit risk using
synthetic loan applicant data. Includes risk scoring, visualizations,
and a simple CLI interface.
"""

import csv
import random
import os
import sys
from datetime import datetime

# ── Try importing optional libraries ─────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ─────────────────────────────────────────────────────────────────────────────
# 1.  RISK SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def calculate_risk_score(applicant: dict) -> dict:
    """
    Score a single applicant (0–100, higher = riskier).

    Factors
    -------
    - Credit score        (higher is better → lowers risk)
    - Debt-to-Income (DTI) ratio (higher → riskier)
    - Employment status   (employed → safer)
    - Loan-to-Value (LTV) ratio (higher → riskier)
    - Payment history     (missed payments → riskier)
    """
    score = 0

    # Credit score component (0–35 pts)
    credit = int(applicant["credit_score"])
    if credit >= 750:
        score += 0
    elif credit >= 700:
        score += 8
    elif credit >= 650:
        score += 16
    elif credit >= 600:
        score += 25
    else:
        score += 35

    # DTI component (0–25 pts)
    dti = float(applicant["dti_ratio"])
    if dti < 0.20:
        score += 0
    elif dti < 0.35:
        score += 10
    elif dti < 0.50:
        score += 18
    else:
        score += 25

    # Employment (0–15 pts)
    emp = applicant["employment_status"].lower()
    if emp == "employed":
        score += 0
    elif emp == "self-employed":
        score += 8
    else:
        score += 15

    # LTV (0–15 pts)
    ltv = float(applicant["ltv_ratio"])
    if ltv <= 0.60:
        score += 0
    elif ltv <= 0.75:
        score += 5
    elif ltv <= 0.90:
        score += 10
    else:
        score += 15

    # Missed payments (0–10 pts)
    missed = int(applicant["missed_payments"])
    score += min(missed * 3, 10)

    # Determine tier
    if score <= 25:
        tier = "LOW"
        color = "green"
    elif score <= 50:
        tier = "MEDIUM"
        color = "orange"
    elif score <= 75:
        tier = "HIGH"
        color = "red"
    else:
        tier = "VERY HIGH"
        color = "darkred"

    return {
        **applicant,
        "risk_score": score,
        "risk_tier": tier,
        "color": color,
    }


def score_portfolio(applicants: list) -> list:
    """Score every applicant in the list."""
    return [calculate_risk_score(a) for a in applicants]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SAMPLE DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_data(n: int = 50, seed: int = 42) -> list:
    """Generate n synthetic loan applicants."""
    random.seed(seed)
    employment_choices = ["employed", "self-employed", "unemployed"]
    data = []
    for i in range(1, n + 1):
        data.append({
            "id": f"APP-{i:04d}",
            "name": f"Applicant {i}",
            "credit_score": random.randint(500, 820),
            "dti_ratio": round(random.uniform(0.10, 0.65), 2),
            "employment_status": random.choices(
                employment_choices, weights=[70, 20, 10]
            )[0],
            "ltv_ratio": round(random.uniform(0.40, 1.05), 2),
            "missed_payments": random.choices(range(6), weights=[50, 20, 12, 8, 6, 4])[0],
            "loan_amount": random.randint(50_000, 5_000_000),
        })
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STATISTICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_stats(scored: list) -> dict:
    scores = [a["risk_score"] for a in scored]
    tiers = [a["risk_tier"] for a in scored]
    total = len(scored)
    avg = sum(scores) / total if total else 0
    tier_counts = {t: tiers.count(t) for t in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]}
    return {
        "total": total,
        "average_score": round(avg, 1),
        "min_score": min(scores),
        "max_score": max(scores),
        "tier_counts": tier_counts,
        "high_risk_pct": round(
            (tier_counts["HIGH"] + tier_counts["VERY HIGH"]) / total * 100, 1
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(scored: list, output_dir: str = "."):
    """Generate and save a 2×2 dashboard of charts."""
    if not HAS_PLOT:
        print("  [!] matplotlib/numpy not installed — skipping charts.")
        print("      Run:  pip install matplotlib numpy")
        return

    stats = portfolio_stats(scored)
    scores = [a["risk_score"] for a in scored]
    credit_scores = [int(a["credit_score"]) for a in scored]
    dti_values = [float(a["dti_ratio"]) for a in scored]
    tiers = stats["tier_counts"]

    tier_labels = list(tiers.keys())
    tier_vals = list(tiers.values())
    tier_colors = ["#27ae60", "#f39c12", "#e74c3c", "#922b21"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Credit Risk Portfolio Dashboard", fontsize=18, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#f8f9fa")

    # ── Chart 1: Risk tier distribution (bar) ────────────────────────────────
    ax1 = axes[0, 0]
    bars = ax1.bar(tier_labels, tier_vals, color=tier_colors, edgecolor="white", linewidth=1.2)
    ax1.set_title("Risk Tier Distribution", fontweight="bold")
    ax1.set_ylabel("Number of Applicants")
    ax1.set_facecolor("#f0f0f0")
    for bar, val in zip(bars, tier_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha="center", va="bottom", fontweight="bold")

    # ── Chart 2: Risk score histogram ────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.hist(scores, bins=15, color="#2980b9", edgecolor="white", linewidth=0.8)
    ax2.axvline(stats["average_score"], color="#e74c3c", linestyle="--",
                linewidth=2, label=f"Avg: {stats['average_score']}")
    ax2.set_title("Risk Score Distribution", fontweight="bold")
    ax2.set_xlabel("Risk Score (0–100)")
    ax2.set_ylabel("Frequency")
    ax2.set_facecolor("#f0f0f0")
    ax2.legend()

    # ── Chart 3: Credit score vs Risk score scatter ───────────────────────────
    ax3 = axes[1, 0]
    color_map = {"LOW": "#27ae60", "MEDIUM": "#f39c12",
                 "HIGH": "#e74c3c", "VERY HIGH": "#922b21"}
    for a in scored:
        ax3.scatter(int(a["credit_score"]), a["risk_score"],
                    color=color_map[a["risk_tier"]], alpha=0.7, s=50)
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax3.legend(handles=patches, fontsize=8)
    ax3.set_title("Credit Score vs Risk Score", fontweight="bold")
    ax3.set_xlabel("Credit Score")
    ax3.set_ylabel("Risk Score")
    ax3.set_facecolor("#f0f0f0")

    # ── Chart 4: DTI vs Risk score scatter ────────────────────────────────────
    ax4 = axes[1, 1]
    for a in scored:
        ax4.scatter(float(a["dti_ratio"]), a["risk_score"],
                    color=color_map[a["risk_tier"]], alpha=0.7, s=50)
    ax4.legend(handles=patches, fontsize=8)
    ax4.set_title("DTI Ratio vs Risk Score", fontweight="bold")
    ax4.set_xlabel("Debt-to-Income Ratio")
    ax4.set_ylabel("Risk Score")
    ax4.set_facecolor("#f0f0f0")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(output_dir, "risk_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Dashboard saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  REPORT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(scored: list, output_dir: str = "."):
    """Write scored results to a CSV file."""
    out_path = os.path.join(output_dir, "scored_applicants.csv")
    fields = ["id", "name", "credit_score", "dti_ratio", "employment_status",
              "ltv_ratio", "missed_payments", "loan_amount", "risk_score", "risk_tier"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for a in scored:
            writer.writerow({k: a[k] for k in fields})
    print(f"  [✓] CSV report saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║        CREDIT RISK ANALYZER  v1.0                   ║
║        Banking & Financial Services Analytics        ║
╚══════════════════════════════════════════════════════╝
"""

MENU = """
  [1]  Score a single applicant
  [2]  Analyze sample portfolio (50 applicants)
  [3]  Load portfolio from CSV file
  [4]  Exit
"""


def prompt_single_applicant() -> dict:
    """Interactively gather one applicant's details."""
    print("\n  Enter applicant details:")
    app_id = input("  Applicant ID (e.g. APP-0001): ").strip() or "APP-0001"
    name = input("  Full name: ").strip() or "Unknown"

    def get_int(prompt, lo, hi):
        while True:
            try:
                v = int(input(f"  {prompt} ({lo}–{hi}): "))
                if lo <= v <= hi:
                    return v
                print(f"  Please enter a value between {lo} and {hi}.")
            except ValueError:
                print("  Invalid — please enter a whole number.")

    def get_float(prompt, lo, hi):
        while True:
            try:
                v = float(input(f"  {prompt} ({lo}–{hi}): "))
                if lo <= v <= hi:
                    return round(v, 2)
                print(f"  Please enter a value between {lo} and {hi}.")
            except ValueError:
                print("  Invalid — please enter a decimal number.")

    credit_score = get_int("Credit score", 300, 850)
    dti = get_float("Debt-to-Income ratio (e.g. 0.35 for 35%)", 0.0, 1.5)
    print("  Employment status options: employed / self-employed / unemployed")
    emp = input("  Employment status: ").strip().lower() or "employed"
    ltv = get_float("Loan-to-Value ratio (e.g. 0.80 for 80%)", 0.0, 1.5)
    missed = get_int("Number of missed payments (last 12 months)", 0, 12)
    loan = get_int("Loan amount (PHP)", 1, 100_000_000)

    return {
        "id": app_id,
        "name": name,
        "credit_score": credit_score,
        "dti_ratio": dti,
        "employment_status": emp,
        "ltv_ratio": ltv,
        "missed_payments": missed,
        "loan_amount": loan,
    }


def print_single_result(result: dict):
    """Pretty-print one applicant's risk result."""
    bar_len = result["risk_score"] // 2
    bar = "█" * bar_len + "░" * (50 - bar_len)
    print(f"""
  ┌─────────────────────────────────────────────┐
  │  Applicant : {result['name']:<30} │
  │  ID        : {result['id']:<30} │
  ├─────────────────────────────────────────────┤
  │  Credit Score     : {result['credit_score']:<24} │
  │  DTI Ratio        : {result['dti_ratio']:<24} │
  │  Employment       : {result['employment_status']:<24} │
  │  LTV Ratio        : {result['ltv_ratio']:<24} │
  │  Missed Payments  : {result['missed_payments']:<24} │
  ├─────────────────────────────────────────────┤
  │  RISK SCORE  : {result['risk_score']:>3} / 100                      │
  │  [{bar}]  │
  │  RISK TIER   : {result['risk_tier']:<30} │
  └─────────────────────────────────────────────┘""")


def print_portfolio_summary(stats: dict):
    print(f"""
  ── Portfolio Summary ──────────────────────────
  Total Applicants  : {stats['total']}
  Average Risk Score: {stats['average_score']}
  Min / Max Score   : {stats['min_score']} / {stats['max_score']}
  High-Risk %       : {stats['high_risk_pct']}%

  Tier Breakdown:
    LOW       : {stats['tier_counts']['LOW']}
    MEDIUM    : {stats['tier_counts']['MEDIUM']}
    HIGH      : {stats['tier_counts']['HIGH']}
    VERY HIGH : {stats['tier_counts']['VERY HIGH']}
  ───────────────────────────────────────────────""")


def load_csv_portfolio(filepath: str) -> list:
    """Load applicants from a CSV file."""
    applicants = []
    required = {"id", "credit_score", "dti_ratio", "employment_status",
                "ltv_ratio", "missed_payments"}
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            missing = required - set(row.keys())
            if missing:
                print(f"  [!] Row {i} missing fields: {missing} — skipped.")
                continue
            row.setdefault("name", f"Applicant {i}")
            row.setdefault("loan_amount", 0)
            applicants.append(row)
    return applicants


def main():
    print(BANNER)
    os.makedirs("output", exist_ok=True)

    while True:
        print(MENU)
        choice = input("  Select an option: ").strip()

        if choice == "1":
            app = prompt_single_applicant()
            result = calculate_risk_score(app)
            print_single_result(result)

        elif choice == "2":
            print("\n  Generating 50 synthetic applicants…")
            data = generate_sample_data(50)
            scored = score_portfolio(data)
            stats = portfolio_stats(scored)
            print_portfolio_summary(stats)
            export_csv(scored, output_dir="output")
            plot_dashboard(scored, output_dir="output")
            print(f"\n  Done! Files saved in ./output/")

        elif choice == "3":
            path = input("\n  Enter path to CSV file: ").strip()
            if not os.path.exists(path):
                print(f"  [!] File not found: {path}")
                continue
            try:
                data = load_csv_portfolio(path)
                if not data:
                    print("  [!] No valid rows found in file.")
                    continue
                scored = score_portfolio(data)
                stats = portfolio_stats(scored)
                print_portfolio_summary(stats)
                export_csv(scored, output_dir="output")
                plot_dashboard(scored, output_dir="output")
            except Exception as e:
                print(f"  [!] Error reading file: {e}")

        elif choice == "4":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid option. Please enter 1–4.")


if __name__ == "__main__":
    main()
