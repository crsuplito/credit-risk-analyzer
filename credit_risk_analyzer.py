"""
Credit Risk Analyzer
====================
A beginner-friendly Python tool for analyzing credit risk using
loan applicant data from credit_risk_dataset.csv.

CSV Fields
----------
person_age              – Applicant age
person_income           – Annual income
person_home_ownership   – RENT / OWN / MORTGAGE / OTHER
person_emp_length       – Employment length in years
loan_intent             – PERSONAL / EDUCATION / MEDICAL / VENTURE /
                          HOMEIMPROVEMENT / DEBTCONSOLIDATION
loan_grade              – A (best) to G (worst)
loan_amnt               – Requested loan amount
loan_int_rate           – Interest rate (%)
loan_status             – 1 = defaulted, 0 = no default
loan_percent_income     – loan_amnt / person_income
cb_person_default_on_file – Y = has prior default, N = none
cb_person_cred_hist_length – Credit history length in years
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

GRADE_SCORES = {"A": 0, "B": 8, "C": 16, "D": 24, "E": 30, "F": 35, "G": 35}

def _safe_float(value, default=0.0):
    """Return float, or default if value is blank/invalid."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def _safe_int(value, default=0):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def calculate_risk_score(applicant: dict) -> dict:
    """
    Score a single applicant (0–100, higher = riskier).

    Factors & weights
    -----------------
    Loan grade (A–F)            0–35 pts  – proxy for creditworthiness
    Loan-to-Income %            0–20 pts  – repayment burden
    Interest rate               0–15 pts  – rate reflects lender's risk view
    Home ownership              0–10 pts  – stability signal
    Prior default on file       0–10 pts  – direct default history
    Employment length           0–5  pts  – job stability
    Credit history length       0–5  pts  – thin file is riskier
    """
    score = 0

    # ── Loan grade (0–35 pts) ────────────────────────────────────────────────
    grade = str(applicant.get("loan_grade", "")).strip().upper()
    score += GRADE_SCORES.get(grade, 35)

    # ── Loan-percent-income / debt burden (0–20 pts) ─────────────────────────
    lpi = _safe_float(applicant.get("loan_percent_income"))
    if lpi < 0.10:
        score += 0
    elif lpi < 0.20:
        score += 5
    elif lpi < 0.35:
        score += 10
    elif lpi < 0.50:
        score += 15
    else:
        score += 20

    # ── Interest rate (0–15 pts) ─────────────────────────────────────────────
    rate = _safe_float(applicant.get("loan_int_rate"))
    if rate == 0.0:
        # Missing rate — assign moderate penalty
        score += 8
    elif rate < 7.0:
        score += 0
    elif rate < 10.0:
        score += 4
    elif rate < 14.0:
        score += 8
    elif rate < 18.0:
        score += 12
    else:
        score += 15

    # ── Home ownership (0–10 pts) ────────────────────────────────────────────
    ownership = str(applicant.get("person_home_ownership", "")).strip().upper()
    ownership_map = {"MORTGAGE": 2, "OWN": 0, "RENT": 6, "OTHER": 10}
    score += ownership_map.get(ownership, 6)

    # ── Prior default on file (0–10 pts) ─────────────────────────────────────
    default_flag = str(applicant.get("cb_person_default_on_file", "N")).strip().upper()
    score += 10 if default_flag == "Y" else 0

    # ── Employment length (0–5 pts) ──────────────────────────────────────────
    emp = _safe_float(applicant.get("person_emp_length"))
    if emp >= 5:
        score += 0
    elif emp >= 2:
        score += 2
    else:
        score += 5

    # ── Credit history length (0–5 pts) ──────────────────────────────────────
    hist = _safe_float(applicant.get("cb_person_cred_hist_length"))
    if hist >= 10:
        score += 0
    elif hist >= 5:
        score += 2
    else:
        score += 5

    # Clamp to 0–100
    score = max(0, min(score, 100))

    # Determine tier
    if score <= 25:
        tier, color = "LOW",       "green"
    elif score <= 50:
        tier, color = "MEDIUM",    "orange"
    elif score <= 75:
        tier, color = "HIGH",      "red"
    else:
        tier, color = "VERY HIGH", "darkred"

    return {**applicant, "risk_score": score, "risk_tier": tier, "color": color}


def score_portfolio(applicants: list) -> list:
    return [calculate_risk_score(a) for a in applicants]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SAMPLE DATA GENERATOR  (matches CSV schema)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample_data(n: int = 50, seed: int = 42) -> list:
    """Generate n synthetic applicants matching the CSV schema."""
    random.seed(seed)
    ownerships  = ["RENT", "OWN", "MORTGAGE", "OTHER"]
    intents     = ["PERSONAL", "EDUCATION", "MEDICAL",
                   "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    grades      = ["A", "B", "C", "D", "E", "F"]
    grade_rates = {"A": (5, 9), "B": (9, 12), "C": (12, 15),
                   "D": (15, 17), "E": (17, 20), "F": (20, 24)}
    data = []
    for i in range(1, n + 1):
        grade = random.choices(grades, weights=[25, 25, 20, 15, 10, 5])[0]
        lo, hi = grade_rates[grade]
        income = random.randint(15_000, 300_000)
        loan   = random.randint(1_000, 35_000)
        data.append({
            "id":                         f"APP-{i:04d}",
            "person_age":                 random.randint(20, 70),
            "person_income":              income,
            "person_home_ownership":      random.choices(
                                              ownerships, weights=[40, 20, 35, 5])[0],
            "person_emp_length":          round(random.uniform(0, 20), 1),
            "loan_intent":                random.choice(intents),
            "loan_grade":                 grade,
            "loan_amnt":                  loan,
            "loan_int_rate":              round(random.uniform(lo, hi), 2),
            "loan_status":                random.choices([0, 1], weights=[75, 25])[0],
            "loan_percent_income":        round(loan / income, 2),
            "cb_person_default_on_file":  random.choices(["N", "Y"], weights=[80, 20])[0],
            "cb_person_cred_hist_length": random.randint(2, 30),
        })
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STATISTICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_stats(scored: list) -> dict:
    scores = [a["risk_score"] for a in scored]
    tiers  = [a["risk_tier"]  for a in scored]
    total  = len(scored)
    avg    = sum(scores) / total if total else 0
    tier_counts = {t: tiers.count(t) for t in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]}

    # Intent breakdown
    intents = [str(a.get("loan_intent", "UNKNOWN")).upper() for a in scored]
    intent_set = sorted(set(intents))
    intent_counts = {i: intents.count(i) for i in intent_set}

    # Grade breakdown
    grades = [str(a.get("loan_grade", "?")).upper() for a in scored]
    grade_counts = {g: grades.count(g) for g in sorted(set(grades))}

    return {
        "total":          total,
        "average_score":  round(avg, 1),
        "min_score":      min(scores),
        "max_score":      max(scores),
        "tier_counts":    tier_counts,
        "intent_counts":  intent_counts,
        "grade_counts":   grade_counts,
        "high_risk_pct":  round(
            (tier_counts["HIGH"] + tier_counts["VERY HIGH"]) / total * 100, 1
        ) if total else 0,
        "default_rate":   round(
            sum(1 for a in scored if str(a.get("loan_status")) == "1") / total * 100, 1
        ) if total else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_dashboard(scored: list, output_dir: str = "."):
    if not HAS_PLOT:
        print("  [!] matplotlib/numpy not installed — skipping charts.")
        print("      Run:  pip install matplotlib numpy")
        return

    stats  = portfolio_stats(scored)
    scores = [a["risk_score"] for a in scored]
    color_map = {
        "LOW":       "#27ae60",
        "MEDIUM":    "#f39c12",
        "HIGH":      "#e74c3c",
        "VERY HIGH": "#922b21",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Credit Risk Portfolio Dashboard", fontsize=18, fontweight="bold", y=0.99)
    fig.patch.set_facecolor("#f8f9fa")

    # ── Chart 1: Risk tier distribution ──────────────────────────────────────
    ax1 = axes[0, 0]
    tiers  = list(stats["tier_counts"].keys())
    tcvals = list(stats["tier_counts"].values())
    tcolors = [color_map[t] for t in tiers]
    bars = ax1.bar(tiers, tcvals, color=tcolors, edgecolor="white", linewidth=1.2)
    ax1.set_title("Risk Tier Distribution", fontweight="bold")
    ax1.set_ylabel("Applicants")
    ax1.set_facecolor("#f0f0f0")
    for bar, val in zip(bars, tcvals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha="center", va="bottom", fontweight="bold")

    # ── Chart 2: Risk score histogram ────────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.hist(scores, bins=20, color="#2980b9", edgecolor="white", linewidth=0.8)
    ax2.axvline(stats["average_score"], color="#e74c3c", linestyle="--",
                linewidth=2, label=f"Avg: {stats['average_score']}")
    ax2.set_title("Risk Score Distribution", fontweight="bold")
    ax2.set_xlabel("Risk Score (0–100)")
    ax2.set_ylabel("Frequency")
    ax2.set_facecolor("#f0f0f0")
    ax2.legend()

    # ── Chart 3: Loan grade vs Risk score (box-ish scatter) ──────────────────
    ax3 = axes[0, 2]
    grade_order = ["A", "B", "C", "D", "E", "F", "G"]
    grade_scores = {g: [] for g in grade_order}
    for a in scored:
        g = str(a.get("loan_grade", "?")).upper()
        if g in grade_scores:
            grade_scores[g].append(a["risk_score"])
    present = [g for g in grade_order if grade_scores[g]]
    positions = range(len(present))
    ax3.boxplot([grade_scores[g] for g in present], positions=list(positions),
                patch_artist=True,
                boxprops=dict(facecolor="#2980b9", alpha=0.6))
    ax3.set_xticks(list(positions))
    ax3.set_xticklabels(present)
    ax3.set_title("Loan Grade vs Risk Score", fontweight="bold")
    ax3.set_xlabel("Loan Grade")
    ax3.set_ylabel("Risk Score")
    ax3.set_facecolor("#f0f0f0")

    # ── Chart 4: Loan-percent-income vs Risk score ────────────────────────────
    ax4 = axes[1, 0]
    for a in scored:
        ax4.scatter(
            _safe_float(a.get("loan_percent_income")),
            a["risk_score"],
            color=color_map[a["risk_tier"]],
            alpha=0.5, s=30,
        )
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    ax4.legend(handles=patches, fontsize=7)
    ax4.set_title("Loan-to-Income % vs Risk Score", fontweight="bold")
    ax4.set_xlabel("Loan Percent of Income")
    ax4.set_ylabel("Risk Score")
    ax4.set_facecolor("#f0f0f0")

    # ── Chart 5: Interest rate vs Risk score ─────────────────────────────────
    ax5 = axes[1, 1]
    for a in scored:
        rate = _safe_float(a.get("loan_int_rate"))
        if rate > 0:
            ax5.scatter(rate, a["risk_score"],
                        color=color_map[a["risk_tier"]], alpha=0.5, s=30)
    ax5.legend(handles=patches, fontsize=7)
    ax5.set_title("Interest Rate vs Risk Score", fontweight="bold")
    ax5.set_xlabel("Interest Rate (%)")
    ax5.set_ylabel("Risk Score")
    ax5.set_facecolor("#f0f0f0")

    # ── Chart 6: Loan intent breakdown (horizontal bar) ───────────────────────
    ax6 = axes[1, 2]
    intent_labels = list(stats["intent_counts"].keys())
    intent_vals   = list(stats["intent_counts"].values())
    colors6 = plt.cm.Set2(range(len(intent_labels)))
    ax6.barh(intent_labels, intent_vals, color=colors6, edgecolor="white")
    ax6.set_title("Loan Intent Breakdown", fontweight="bold")
    ax6.set_xlabel("Applicants")
    ax6.set_facecolor("#f0f0f0")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(output_dir, "risk_dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Dashboard saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  REPORT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

EXPORT_FIELDS = [
    "id", "person_age", "person_income", "person_home_ownership",
    "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
    "loan_int_rate", "loan_status", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
    "risk_score", "risk_tier",
]

def export_csv(scored: list, output_dir: str = "."):
    out_path = os.path.join(output_dir, "scored_applicants.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for a in scored:
            row = {k: a.get(k, "") for k in EXPORT_FIELDS}
            # Synthetic data has 'id'; CSV rows may not — fall back to row index
            if not row["id"]:
                row["id"] = a.get("_row", "")
            writer.writerow(row)
    print(f"  [✓] CSV report saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║        CREDIT RISK ANALYZER  v1.0                    ║
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
    print("\n  Enter applicant details:")
    app_id = input("  Applicant ID (e.g. APP-0001): ").strip() or "APP-0001"

    def get_int(prompt, lo, hi):
        while True:
            try:
                v = int(input(f"  {prompt} ({lo}–{hi}): "))
                if lo <= v <= hi:
                    return v
                print(f"  Please enter a value between {lo} and {hi}.")
            except ValueError:
                print("  Invalid — please enter a whole number.")

    def get_float(prompt, lo, hi, allow_blank=False):
        while True:
            raw = input(f"  {prompt} ({lo}–{hi}){' [Enter to skip]' if allow_blank else ''}: ").strip()
            if allow_blank and raw == "":
                return ""
            try:
                v = float(raw)
                if lo <= v <= hi:
                    return round(v, 2)
                print(f"  Please enter a value between {lo} and {hi}.")
            except ValueError:
                print("  Invalid — please enter a decimal number.")

    def get_choice(prompt, options):
        opts_str = " / ".join(options)
        while True:
            v = input(f"  {prompt} ({opts_str}): ").strip().upper()
            if v in options:
                return v
            print(f"  Please choose one of: {opts_str}")

    age         = get_int("Age", 18, 100)
    income      = get_int("Annual income", 1, 100_000_000)
    ownership   = get_choice("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    emp_length  = get_float("Employment length in years", 0, 60)
    print("  Loan intent options: PERSONAL / EDUCATION / MEDICAL / "
          "VENTURE / HOMEIMPROVEMENT / DEBTCONSOLIDATION")
    intent      = get_choice("Loan intent",
                             ["PERSONAL", "EDUCATION", "MEDICAL",
                              "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    grade       = get_choice("Loan grade", ["A", "B", "C", "D", "E", "F"])
    loan_amnt   = get_int("Loan amount", 1, 100_000_000)
    int_rate    = get_float("Interest rate % (e.g. 12.5)", 0, 50, allow_blank=True)
    loan_pct    = round(loan_amnt / income, 4) if income else 0
    default_fl  = get_choice("Prior default on file", ["Y", "N"])
    cred_hist   = get_float("Credit history length in years", 0, 60)
    loan_status = get_choice("Current loan status (1=default, 0=no default)", ["0", "1"])

    return {
        "id":                         app_id,
        "person_age":                 age,
        "person_income":              income,
        "person_home_ownership":      ownership,
        "person_emp_length":          emp_length,
        "loan_intent":                intent,
        "loan_grade":                 grade,
        "loan_amnt":                  loan_amnt,
        "loan_int_rate":              int_rate,
        "loan_status":                loan_status,
        "loan_percent_income":        loan_pct,
        "cb_person_default_on_file":  default_fl,
        "cb_person_cred_hist_length": cred_hist,
    }


def print_single_result(result: dict):
    bar_len = result["risk_score"] * 30 // 100
    bar = "█" * bar_len + "░" * (30 - bar_len)
    name_or_id = result.get("id", "N/A")
    print(f"""
  ┌──────────────────────────────────────────────────┐
  │  Applicant ID     : {str(name_or_id):<29}│
  ├──────────────────────────────────────────────────┤
  │  Age              : {str(result.get('person_age','')):<29}│
  │  Annual Income    : {str(result.get('person_income','')):<29}│
  │  Home Ownership   : {str(result.get('person_home_ownership','')):<29}│
  │  Employment Yrs   : {str(result.get('person_emp_length','')):<29}│
  │  Loan Intent      : {str(result.get('loan_intent','')):<29}│
  │  Loan Grade       : {str(result.get('loan_grade','')):<29}│
  │  Loan Amount      : {str(result.get('loan_amnt','')):<29}│
  │  Interest Rate %  : {str(result.get('loan_int_rate','')):<29}│
  │  Loan % of Income : {str(result.get('loan_percent_income','')):<29}│
  │  Prior Default    : {str(result.get('cb_person_default_on_file','')):<29}│
  │  Credit Hist Yrs  : {str(result.get('cb_person_cred_hist_length','')):<29}│
  ├──────────────────────────────────────────────────┤
  │  RISK SCORE  : {result['risk_score']:>3} / 100                      │
  │  [{bar}]                   │
  │  RISK TIER   : {result['risk_tier']:<33}│
  └──────────────────────────────────────────────────┘""")


def print_portfolio_summary(stats: dict):
    grade_str  = "  ".join(f"{g}:{v}" for g, v in sorted(stats["grade_counts"].items()))
    intent_str = "\n".join(f"    {k:<20}: {v}" for k, v in sorted(stats["intent_counts"].items()))
    print(f"""
  ── Portfolio Summary ──────────────────────────────
  Total Applicants  : {stats['total']}
  Average Risk Score: {stats['average_score']}
  Min / Max Score   : {stats['min_score']} / {stats['max_score']}
  High-Risk %       : {stats['high_risk_pct']}%
  Actual Default %  : {stats['default_rate']}%

  Risk Tier Breakdown:
    LOW       : {stats['tier_counts']['LOW']}
    MEDIUM    : {stats['tier_counts']['MEDIUM']}
    HIGH      : {stats['tier_counts']['HIGH']}
    VERY HIGH : {stats['tier_counts']['VERY HIGH']}

  Loan Grade Breakdown:
    {grade_str}

  Loan Intent Breakdown:
{intent_str}
  ───────────────────────────────────────────────────""")


def load_csv_portfolio(filepath: str) -> list:
    """Load applicants from a CSV file matching the credit_risk_dataset schema."""
    required = {
        "person_age", "person_income", "person_home_ownership",
        "loan_grade", "loan_amnt", "loan_percent_income",
        "cb_person_default_on_file", "cb_person_cred_hist_length",
    }
    applicants = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            # Strip whitespace from keys (handles BOM or extra spaces)
            row = {k.strip(): v.strip() for k, v in row.items()}
            missing = required - set(row.keys())
            if missing:
                print(f"  [!] Row {i} missing fields: {missing} — skipped.")
                continue
            row.setdefault("id", f"ROW-{i:05d}")
            row["_row"] = i
            applicants.append(row)
    return applicants


def main():
    print(BANNER)
    os.makedirs("output", exist_ok=True)

    while True:
        print(MENU)
        choice = input("  Select an option: ").strip()

        if choice == "1":
            app    = prompt_single_applicant()
            result = calculate_risk_score(app)
            print_single_result(result)

        elif choice == "2":
            print("\n  Generating 50 synthetic applicants…")
            data   = generate_sample_data(50)
            scored = score_portfolio(data)
            stats  = portfolio_stats(scored)
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
                print(f"\n  Loaded {len(data)} applicants. Scoring…")
                scored = score_portfolio(data)
                stats  = portfolio_stats(scored)
                print_portfolio_summary(stats)
                export_csv(scored, output_dir="output")
                plot_dashboard(scored, output_dir="output")
                print(f"\n  Done! Files saved in ./output/")
            except Exception as e:
                print(f"  [!] Error reading file: {e}")

        elif choice == "4":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid option. Please enter 1–4.")


if __name__ == "__main__":
    main()