# symulacje.py
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1) CONFIG (skalibrowane z pilota)
# ==========================================

# Populacja / czas
NUM_AGENTS = 1000
ITERATIONS = 40
SEED = 123

# Start populacji: pre_scaled (-1..1)
OPINION_MEAN = 0.5357
OPINION_SD   = 0.55     # stabilizacja (mniej clippingu niż 0.6344)

# TRUST (0..1) z pilota (stabilizacja SD)
TRUST_AI_MEAN = 0.5469
TRUST_AI_SD   = 0.25

TRUST_HUMAN_MEAN = 0.5558
TRUST_HUMAN_SD   = 0.28

# Latitude of Acceptance (dist 0..2) – z pilota
LAT_ACCEPT_AI_MEAN = 1.3471
LAT_ACCEPT_AI_SD   = 0.5557

LAT_ACCEPT_HUMAN_MEAN = 1.6900
LAT_ACCEPT_HUMAN_SD   = 0.3642

# Latitude of Rejection – stabilne: Lr = La + buffer
REJECT_BUFFER_AI = 0.40
REJECT_BUFFER_H  = 0.45

# Boomerang strength (0..1) – wersja "high dist"
BOOM_AI_MEAN = 0.6176
BOOM_AI_SD   = 0.27

BOOM_HUMAN_MEAN = 0.5809
BOOM_HUMAN_SD   = 0.34

# Susceptibility (0..1): Beta (zamiast uniform – stabilniej i ładniej)
# Beta(2,2) daje większość w środku, mniej ekstremów
SUSC_ALPHA = 2.0
SUSC_BETA  = 2.0

# Lista bodźców R (Twoje wagi N1..N16)
R_ITEMS = np.array([
    0.00, 0.03, -0.03, 0.00,   # N1..N4
    -0.30, -0.30, 0.30, 0.30,  # N5..N8
    -0.80, -0.80, -0.87, -0.93,# N9..N12
    1.00, 1.00, 0.93, 0.87     # N13..N16
], dtype=float)

"""# Scenariusz feedu: 'neutral' | 'polarizing_right' | 'polarizing_left'"""
FEED_MODE = "neutral"

"""# Źródło: 'AI' lub 'HUMAN'"""
SOURCE_TYPE = "HUMAN"  # <- zmień na "HUMAN" albo zrób pętlę porównawczą

# ==========================================
# 2) Pomocnicze funkcje losowania z clippingiem
# ==========================================

def clip(x, lo, hi):
    return float(np.clip(x, lo, hi))

def sample_normal_clipped(mean, sd, lo, hi, rng):
    return clip(rng.normal(mean, sd), lo, hi)

def sample_beta_clipped(alpha, beta, rng):
    return clip(rng.beta(alpha, beta), 0.0, 1.0)

def sample_latitudes(source_type, rng):
    """Losuje (La, Lr, boom) dla agenta zależnie od źródła."""
    if source_type == "AI":
        La = sample_normal_clipped(LAT_ACCEPT_AI_MEAN, LAT_ACCEPT_AI_SD, 0.0, 2.0, rng)
        Lr = clip(La + REJECT_BUFFER_AI, 0.0, 2.0)
        boom = sample_normal_clipped(BOOM_AI_MEAN, BOOM_AI_SD, 0.0, 1.0, rng)
    else:
        La = sample_normal_clipped(LAT_ACCEPT_HUMAN_MEAN, LAT_ACCEPT_HUMAN_SD, 0.0, 2.0, rng)
        Lr = clip(La + REJECT_BUFFER_H, 0.0, 2.0)
        boom = sample_normal_clipped(BOOM_HUMAN_MEAN, BOOM_HUMAN_SD, 0.0, 1.0, rng)

    # Wymuś sensowność: La < Lr (z małym marginesem)
    if La >= Lr:
        mid = (La + Lr) / 2.0
        La = max(0.0, mid - 0.2)
        Lr = min(2.0, mid + 0.2)

    return La, Lr, boom

def sample_trust(source_type, rng):
    if source_type == "AI":
        return sample_normal_clipped(TRUST_AI_MEAN, TRUST_AI_SD, 0.0, 1.0, rng)
    else:
        return sample_normal_clipped(TRUST_HUMAN_MEAN, TRUST_HUMAN_SD, 0.0, 1.0, rng)

# ==========================================
# 3) Generator bodźców (feed)
# ==========================================

def sample_recommendation(feed_mode, rng):
    """
    Zwraca rekomendację w skali -1..+1.
    - neutral: losowo z R_ITEMS
    - polarizing_right: częściej dodatnie skrajności
    - polarizing_left: częściej ujemne skrajności
    """
    if feed_mode == "neutral":
        return float(rng.choice(R_ITEMS))

    # rozdzielamy na "skrajne" i "umiarkowane/neutralne"
    extreme_pos = np.array([1.00, 0.93, 0.87, 0.80], dtype=float)
    extreme_neg = np.array([-0.93, -0.87, -0.80], dtype=float)
    moderate = np.array([-0.30, -0.03, 0.00, 0.03, 0.30], dtype=float)

    if feed_mode == "polarizing_right":
        # 70% skrajne dodatnie, 30% reszta
        if rng.random() < 0.70:
            return float(rng.choice(extreme_pos))
        return float(rng.choice(moderate))

    if feed_mode == "polarizing_left":
        # 70% skrajne ujemne, 30% reszta
        if rng.random() < 0.70:
            return float(rng.choice(extreme_neg))
        return float(rng.choice(moderate))

    raise ValueError(f"Nieznany FEED_MODE: {feed_mode}")

# ==========================================
# 4) Silnik ABM
# ==========================================

class Agent:
    def __init__(self, source_type, rng):
        # start opinii
        self.opinion = sample_normal_clipped(OPINION_MEAN, OPINION_SD, -1.0, 1.0, rng)

        # podatność
        self.susceptibility = sample_beta_clipped(SUSC_ALPHA, SUSC_BETA, rng)  # 0..1

        # zaufanie do danego źródła
        self.trust = sample_trust(source_type, rng)  # 0..1

        # progi + boomerang
        self.lat_acceptance, self.lat_rejection, self.boomerang_strength = sample_latitudes(source_type, rng)

    def update(self, recommendation):
        # Siła wpływu: zaufanie * podatność
        influence = self.trust * self.susceptibility

        # dystans w osi opinii (-1..1), ale progi są w "dist" 0..2
        dist_signed = recommendation - self.opinion
        dist = abs(dist_signed)

        # 1) Strefa akceptacji (asymilacja): zbliżamy się proporcjonalnie do dystansu
        if dist < self.lat_acceptance:
            change = influence * dist_signed

        # 2) Strefa odrzucenia (kontrast + boomerang): „odbijamy się”
        elif dist > self.lat_rejection:
            # Uciekamy w przeciwną stronę względem rekomendacji
            # (znak przeciwny do dist_signed)
            change = -influence * self.boomerang_strength * np.sign(dist_signed)

        # 3) Strefa obojętności: brak zmiany
        else:
            change = 0.0

        # Aktualizacja + clipping
        self.opinion = clip(self.opinion + change, -1.0, 1.0)

def run_simulation(source_type, feed_mode, seed=123):
    rng = np.random.default_rng(seed)

    agents = [Agent(source_type, rng) for _ in range(NUM_AGENTS)]
    opinions_start = np.array([a.opinion for a in agents], dtype=float)

    mean_history = []
    sd_history = []

    for _ in range(ITERATIONS):
        opinions = np.array([a.opinion for a in agents], dtype=float)
        mean_history.append(float(opinions.mean()))
        sd_history.append(float(opinions.std(ddof=1)))

        # Każdy agent dostaje bodziec (tu: jeden bodziec na iterację, wspólny feed)
        rec = sample_recommendation(feed_mode, rng)
        for a in agents:
            a.update(rec)

    opinions_end = np.array([a.opinion for a in agents], dtype=float)
    return opinions_start, opinions_end, np.array(mean_history), np.array(sd_history)

# ==========================================
# 5) Prosta KDE bez seaborn (żeby nie wymagać seaborn)
# ==========================================

def kde_1d(x, grid, bw=0.12):
    # Gaussian KDE (prosta implementacja)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.zeros_like(grid)
    diff = grid[:, None] - x[None, :]
    kern = np.exp(-0.5 * (diff / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    return kern.mean(axis=1)

# ==========================================
# 6) Uruchomienie + wykresy
# ==========================================

if __name__ == "__main__":
    print(f"Start ABM: SOURCE_TYPE={SOURCE_TYPE}, FEED_MODE={FEED_MODE}")
    start, end, mean_hist, sd_hist = run_simulation(SOURCE_TYPE, FEED_MODE, seed=SEED)

    # Wykres 1: średnia opinia w czasie
    plt.figure(figsize=(7,4))
    plt.plot(mean_hist, linewidth=2)
    plt.ylim(-1.05, 1.05)
    plt.xlabel("Iteracja (czas)")
    plt.ylabel("Średnia opinia (-1..+1)")
    plt.title(f"Średnia opinia w czasie | {SOURCE_TYPE} | feed={FEED_MODE}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Wykres 2: SD opinii w czasie (proxy polaryzacji)
    plt.figure(figsize=(7,4))
    plt.plot(sd_hist, linewidth=2)
    plt.xlabel("Iteracja (czas)")
    plt.ylabel("SD opinii (polaryzacja)")
    plt.title(f"Polaryzacja (SD) w czasie | {SOURCE_TYPE} | feed={FEED_MODE}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Wykres 3: rozkład start vs koniec (KDE)
    grid = np.linspace(-1.0, 1.0, 400)
    d0 = kde_1d(start, grid, bw=0.12)
    d1 = kde_1d(end, grid, bw=0.12)

    plt.figure(figsize=(7,4))
    plt.plot(grid, d0, linewidth=2, label="Start")
    plt.plot(grid, d1, linewidth=2, label=f"Koniec (T={ITERATIONS})")
    plt.xlabel("Opinia (-1..+1)")
    plt.ylabel("Gęstość (KDE)")
    plt.title(f"Rozkład opinii | {SOURCE_TYPE} | feed={FEED_MODE}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()