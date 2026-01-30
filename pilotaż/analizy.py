import pandas as pd
import numpy as np

# =========================
# SETTINGS
# =========================
CSV_PATH = "pilotaż\long_ready.csv"
DIST_HIGH = 1.2               # co uznajemy za "duży dystans" (do kalibracji boomerangu)

# progi do wyciągania La/Lr (jeśli chcesz liczyć z danych per osoba)
LA_THRESHOLD = 4
LR_THRESHOLD = 3  # uwaga: 4 zwykle za ostre w pilotażu

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_PATH)

# Basic sanity
required_cols = {"id", "source_type", "pre_scaled", "dist", "R", "LA", "LR", "B", "T"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Brakuje kolumn w long_ready.csv: {missing}")

# Ensure numeric for key columns
for col in ["pre_scaled", "dist", "R", "LA", "LR", "B", "T"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 1) Opinion distribution (start populacji)
# =========================
pre_by_person = df.groupby("id")["pre_scaled"].first()  # jedna wartość pre na osobę
OPINION_MEAN = float(pre_by_person.mean())
OPINION_SD = float(pre_by_person.std(ddof=1))

# =========================
# 2) TRUST (z Twojego pytania wiarygodności "R")
#    R jest na skali 1..5 -> skalujemy do 0..1
# =========================
df["trust01"] = (df["R"] - 1.0) / 4.0
trust_stats = df.groupby("source_type")["trust01"].agg(["mean", "std", "count"]).reset_index()

# =========================
# 3) BOOMERANG (z Twojej skali B) -> 0..1
# =========================
df["boom01"] = (df["B"] - 1.0) / 4.0

boom_stats_all = df.groupby("source_type")["boom01"].agg(["mean", "std", "count"]).reset_index()

df_high = df[df["dist"] >= DIST_HIGH].copy()
boom_stats_high = df_high.groupby("source_type")["boom01"].agg(["mean", "std", "count"]).reset_index()

# =========================
# 4) La/Lr progi per osoba (opcjonalnie, ale przydatne)
#    La_i = max(dist | LA >= LA_THRESHOLD)
#    Lr_i = min(dist | LR >= LR_THRESHOLD)
# =========================
def thresholds_per_person(sub):
    rows = []
    for pid, g in sub.groupby("id"):
        la = g.loc[g["LA"] >= LA_THRESHOLD, "dist"].dropna()
        lr = g.loc[g["LR"] >= LR_THRESHOLD, "dist"].dropna()

        La_i = float(la.max()) if len(la) else np.nan
        Lr_i = float(lr.min()) if len(lr) else np.nan
        rows.append({"id": pid, "La": La_i, "Lr": Lr_i})
    return pd.DataFrame(rows)

thr_AI = thresholds_per_person(df[df["source_type"] == "AI"])
thr_H  = thresholds_per_person(df[df["source_type"] == "HUMAN"])

def summarize_thr(thr):
    La = thr["La"].dropna()
    Lr = thr["Lr"].dropna()
    return {
        "La_mean": float(La.mean()) if len(La) else np.nan,
        "La_sd": float(La.std(ddof=1)) if len(La) > 1 else np.nan,
        "Lr_mean": float(Lr.mean()) if len(Lr) else np.nan,
        "Lr_sd": float(Lr.std(ddof=1)) if len(Lr) > 1 else np.nan,
        "n_La": int(len(La)),
        "n_Lr": int(len(Lr)),
    }

thr_AI_sum = summarize_thr(thr_AI)
thr_H_sum  = summarize_thr(thr_H)

# =========================
# PRINT RESULTS
# =========================
print("\n=== 1) OPINION (pre_scaled) ===")
print(f"Participants N = {pre_by_person.shape[0]}")
print(f"OPINION_MEAN = {OPINION_MEAN:.4f}")
print(f"OPINION_SD   = {OPINION_SD:.4f}")

print("\n=== 2) TRUST (R -> trust01) by source_type ===")
print(trust_stats.to_string(index=False))

print("\n=== 3) BOOMERANG (B -> boom01) ALL by source_type ===")
print(boom_stats_all.to_string(index=False))

print(f"\n=== 3b) BOOMERANG (B -> boom01) for dist >= {DIST_HIGH} by source_type ===")
print(boom_stats_high.to_string(index=False))

print("\n=== 4) Thresholds (La/Lr) per source_type ===")
print("AI:", thr_AI_sum)
print("H :", thr_H_sum)

# =========================
# BUILD CONFIG BLOCK (do wklejenia do ABM)
# =========================
def get_row(stats_df, src):
    row = stats_df[stats_df["source_type"] == src]
    if row.empty:
        return np.nan, np.nan
    return float(row["mean"].iloc[0]), float(row["std"].iloc[0] if not pd.isna(row["std"].iloc[0]) else 0.0)

TRUST_AI_MEAN, TRUST_AI_SD = get_row(trust_stats, "AI")
TRUST_H_MEAN,  TRUST_H_SD  = get_row(trust_stats, "HUMAN")

BOOM_AI_MEAN_ALL, BOOM_AI_SD_ALL = get_row(boom_stats_all, "AI")
BOOM_H_MEAN_ALL,  BOOM_H_SD_ALL  = get_row(boom_stats_all, "HUMAN")

BOOM_AI_MEAN_HI, BOOM_AI_SD_HI = get_row(boom_stats_high, "AI")
BOOM_H_MEAN_HI,  BOOM_H_SD_HI  = get_row(boom_stats_high, "HUMAN")

print("\n\n=============================")
print("SUGGESTED CONFIG (paste into ABM)")
print("=============================")
print(f"OPINION_MEAN = {OPINION_MEAN:.4f}   # pre_scaled mean (-1..1)")
print(f"OPINION_SD   = {OPINION_SD:.4f}   # pre_scaled sd")

print(f"\n# TRUST (0..1) from credibility R")
print(f"TRUST_AI_MEAN = {TRUST_AI_MEAN:.4f}")
print(f"TRUST_AI_SD   = {TRUST_AI_SD:.4f}")
print(f"TRUST_HUMAN_MEAN = {TRUST_H_MEAN:.4f}")
print(f"TRUST_HUMAN_SD   = {TRUST_H_SD:.4f}")

print(f"\n# LATITUDES from thresholds (dist scale 0..2)")
print(f"LAT_ACCEPT_AI_MEAN = {thr_AI_sum['La_mean']:.4f}  # n={thr_AI_sum['n_La']}")
print(f"LAT_ACCEPT_AI_SD   = {thr_AI_sum['La_sd']:.4f}")
print(f"LAT_ACCEPT_HUMAN_MEAN = {thr_H_sum['La_mean']:.4f}  # n={thr_H_sum['n_La']}")
print(f"LAT_ACCEPT_HUMAN_SD   = {thr_H_sum['La_sd']:.4f}")

print(f"\n# NOTE: Lr thresholds from LR>= {LR_THRESHOLD} may be unstable in small N.")
print(f"LAT_REJECT_AI_MEAN_raw = {thr_AI_sum['Lr_mean']:.4f}  # n={thr_AI_sum['n_Lr']}")
print(f"LAT_REJECT_AI_SD_raw   = {thr_AI_sum['Lr_sd']:.4f}")
print(f"LAT_REJECT_HUMAN_MEAN_raw = {thr_H_sum['Lr_mean']:.4f}  # n={thr_H_sum['n_Lr']}")
print(f"LAT_REJECT_HUMAN_SD_raw   = {thr_H_sum['Lr_sd']:.4f}")

print(f"\n# BOOMERANG (0..1) from B")
print(f"BOOM_AI_MEAN_ALL = {BOOM_AI_MEAN_ALL:.4f}")
print(f"BOOM_AI_SD_ALL   = {BOOM_AI_SD_ALL:.4f}")
print(f"BOOM_HUMAN_MEAN_ALL = {BOOM_H_MEAN_ALL:.4f}")
print(f"BOOM_HUMAN_SD_ALL   = {BOOM_H_SD_ALL:.4f}")

print(f"\n# BOOMERANG for high dist (dist >= {DIST_HIGH})")
print(f"BOOM_AI_MEAN_HIGH = {BOOM_AI_MEAN_HI:.4f}")
print(f"BOOM_AI_SD_HIGH   = {BOOM_AI_SD_HI:.4f}")
print(f"BOOM_HUMAN_MEAN_HIGH = {BOOM_H_MEAN_HI:.4f}")
print(f"BOOM_HUMAN_SD_HIGH   = {BOOM_H_SD_HI:.4f}")
