import pandas as pd
import numpy as np

# -----------------------------
# 0) Wczytanie danych
# -----------------------------
df = pd.read_csv(
    r"C:\Users\liebe\Desktop\GitHub\doktorat\pilotaż\Dobre_dane.csv"
)


# -----------------------------
# 1) Wagi (R dla bodźców N1..N16)
# -----------------------------
weights = {
    "N1": 0.00,
    "N2": 0.03,
    "N3": -0.03,
    "N4": 0.00,
    "N5": -0.30,
    "N6": -0.30,
    "N7": 0.30,
    "N8": 0.30,
    "N9": -0.80,
    "N10": -0.80,
    "N11": -0.87,
    "N12": -0.93,
    "N13": 1.00,
    "N14": 1.00,
    "N15": 0.93,
    "N16": 0.87,
}

# -----------------------------
# 2) Mapowanie postawy pre/post (jeśli masz teksty)
#    -> skala 1..5
# -----------------------------
LIKERT_MAP = {
    "Zdecydowanie się nie zgadzam": 1,
    "Raczej się nie zgadzam": 2,
    "Ani się nie zgadzam, ani się zgadzam": 3,
    "Raczej się zgadzam": 4,
    "Zdecydowanie się zgadzam": 5,
}

def to_numeric_likert(series: pd.Series) -> pd.Series:
    """
    Jeśli seria jest liczbowa -> zostawia.
    Jeśli tekstowa -> mapuje wg LIKERT_MAP.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.map(LIKERT_MAP).astype(float)

def likert_1to5_to_minus1to1(x: pd.Series) -> pd.Series:
    """
    1..5 -> -1..+1
    1 -> -1, 3 -> 0, 5 -> +1
    """
    return (x - 3.0) / 2.0

# -----------------------------
# 3) Przygotuj PRE/POST
# -----------------------------
df = df.copy()

df["pre_num"]  = to_numeric_likert(df["pre_test_abortion"])
df["post_num"] = to_numeric_likert(df["post_test_abortion"])

# Jeśli pre/post już masz w -1..1 (np. z innego kodowania), to zakomentuj linię poniżej:
df["pre_scaled"]  = likert_1to5_to_minus1to1(df["pre_num"])
df["post_scaled"] = likert_1to5_to_minus1to1(df["post_num"])

# -----------------------------
# 4) Wide -> Long dla kolumn N*_H_* i N*_A_*
# -----------------------------
# wybieramy wszystkie kolumny bodźców
stim_cols = [c for c in df.columns if c.startswith("N") and ("_H_" in c or "_A_" in c)]

# melt do długiego
long = df.melt(
    id_vars=["id", "pre_scaled", "post_scaled", "Gender", "Age", "education", "manipulation_check"],
    value_vars=stim_cols,
    var_name="var",
    value_name="value"
)

# wyciągamy item/source/measure z nazwy kolumny
# format: N12_H_LR
extracted = long["var"].str.extract(r"^(N\d{1,2})_([HA])_(R|LA|LR|B|T)$")
long["item"] = extracted[0]
long["source"] = extracted[1]   # H lub A
long["measure"] = extracted[2]  # R/LA/LR/B/T

# pivot: jedna linia = id x item x source
long = (
    long.pivot_table(
        index=["id", "item", "source", "pre_scaled", "post_scaled", "Gender", "Age", "education", "manipulation_check"],
        columns="measure",
        values="value",
        aggfunc="first"
    )
    .reset_index()
)

# porządkujemy nazwy kolumn (R/LA/LR/B/T to Twoje odpowiedzi)
long.columns.name = None

# -----------------------------
# 5) Doklej wagę bodźca i policz dist
# -----------------------------
long["R_item"] = long["item"].map(weights).astype(float)

# dist liczymy od postawy bazowej (pre)
long["dist"] = (long["R_item"] - long["pre_scaled"]).abs()

# czytelniejsze nazwy źródła
long["source_type"] = long["source"].map({"H": "HUMAN", "A": "AI"})

# (opcjonalnie) upewnij się, że odpowiedzi R/LA/LR/B/T są liczbowe:
for col in ["R", "LA", "LR", "B", "T"]:
    if col in long.columns:
        long[col] = pd.to_numeric(long[col], errors="coerce")

print(long.head())
print("Wymiary long:", long.shape)

# -----------------------------
# 6) (opcjonalnie) zapis do pliku
# -----------------------------
long.to_csv("long_ready.csv", index=False)

