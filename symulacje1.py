import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETRY Z PILOTAŻU
# =========================

OPINION_MEAN = 0.5357
OPINION_SD   = 0.6344

TRUST = {
    "AI":    (0.5469, 0.3085),
    "HUMAN": (0.5558, 0.3547)
}

LAT_ACCEPT = {
    "AI":    (1.3471, 0.5557),
    "HUMAN": (1.6900, 0.3642)
}

LAT_REJECT = {
    "AI":    (0.1143, 0.1069),
    "HUMAN": (0.0217, 0.0531)
}

BOOM = {
    "AI":    (0.6176, 0.2699),
    "HUMAN": (0.5809, 0.3411)
}

# =========================
# FEED
# =========================

R_ITEMS = np.array([
    0.00, 0.03, -0.03, 0.00,
    -0.30, -0.30, 0.30, 0.30,
    -0.80, -0.80, -0.87, -0.93,
    1.00, 1.00, 0.93, 0.87
])

def sample_R(feed="neutral"):
    return float(np.random.choice(R_ITEMS))

# =========================
# AGENT (LAT)
# =========================

class Agent:
    def __init__(self, source):
        self.opinion = np.clip(
            np.random.normal(OPINION_MEAN, OPINION_SD), -1, 1
        )

        self.trust = np.clip(
            np.random.normal(*TRUST[source]), 0, 1
        )

        self.lat_accept = np.clip(
            np.random.normal(*LAT_ACCEPT[source]), 0.05, 2.0
        )

        # LAT REJECT > LAT ACCEPT (ważne!)
        self.lat_reject = np.clip(
            self.lat_accept + abs(np.random.normal(*LAT_REJECT[source])),
            self.lat_accept + 0.05,
            2.0
        )

        self.boom = np.clip(
            np.random.normal(*BOOM[source]), 0, 1
        )

    def update(self, R):
        dist = abs(R - self.opinion)
        delta = R - self.opinion

        # ASYMILACJA
        if dist <= self.lat_accept:
            self.opinion += self.trust * delta

        # KONTRAST / BOOMERANG
        elif dist >= self.lat_reject:
            self.opinion -= self.trust * self.boom * np.sign(delta)

        # OBOJĘTNOŚĆ: nic się nie dzieje

        self.opinion = np.clip(self.opinion, -1, 1)

# =========================
# SYMULACJA
# =========================

def run_simulation(source="AI", N=500, T=16, seed=123):
    np.random.seed(seed)

    agents = [Agent(source) for _ in range(N)]
    start = np.array([a.opinion for a in agents])

    mean_hist, sd_hist = [], []

    for _ in range(T):
        opinions = np.array([a.opinion for a in agents])
        mean_hist.append(opinions.mean())
        sd_hist.append(opinions.std())

        R = sample_R()
        for a in agents:
            a.update(R)

    end = np.array([a.opinion for a in agents])
    return start, end, np.array(mean_hist), np.array(sd_hist)

#=== Replikacja
def replicate_simulation(source="AI", R=30, N=500, T=16, seed=123):
    all_means = []
    all_sds   = []
    starts    = []
    ends      = []

    for r in range(R):
        start, end, mean_hist, sd_hist = run_simulation(
            source=source,
            N=N,
            T=T,
            seed=seed + r
        )
        all_means.append(mean_hist)
        all_sds.append(sd_hist)
        starts.append(start)
        ends.append(end)

    return {
        "mean": np.array(all_means),   # (R, T)
        "sd":   np.array(all_sds),     # (R, T)
        "start": np.array(starts),     # (R, N)
        "end":   np.array(ends)        # (R, N)
    }


# =========================
# WYKRESY
# =========================

def plot_all(start, end, mean_hist, sd_hist, source):
    # mean
    plt.figure(figsize=(7,4))
    plt.plot(mean_hist)
    plt.ylim(-1,1)
    plt.title(f"Średnia opinia | {source}")
    plt.xlabel("Iteracja")
    plt.ylabel("Średnia opinia")
    plt.grid(alpha=0.3)
    plt.show()

    # sd
    plt.figure(figsize=(7,4))
    plt.plot(sd_hist)
    plt.title(f"Polaryzacja (SD) | {source}")
    plt.xlabel("Iteracja")
    plt.ylabel("SD")
    plt.grid(alpha=0.3)
    plt.show()

    # histogram
    bins = np.linspace(-1, 1, 40)
    plt.figure(figsize=(7,4))
    plt.hist(start, bins=bins, density=True, alpha=0.6, label="Start")
    plt.hist(end, bins=bins, density=True, alpha=0.6, label="Koniec")
    plt.title(f"Rozkład opinii | {source}")
    plt.xlabel("Opinia")
    plt.ylabel("Gęstość")
    plt.legend()
    plt.show()

# =========================
# PORÓWNANIE AI vs HUMAN
# =========================

def plot_compare(ai, human):
    start_ai, end_ai, mean_ai, sd_ai = ai
    start_h,  end_h,  mean_h,  sd_h  = human

    T = len(mean_ai)

    # --- ŚREDNIA OPINIA ---
    plt.figure(figsize=(7,4))
    plt.plot(mean_ai, label="AI", color="tab:blue")
    plt.plot(mean_h, label="Human", color="tab:orange")
    plt.axhline(0, color="grey", linestyle="--", alpha=0.5)
    plt.ylim(-1, 1)
    plt.xlabel("Iteracja")
    plt.ylabel("Średnia opinia")
    plt.title("Średnia opinia w czasie")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # --- POLARYZACJA ---
    plt.figure(figsize=(7,4))
    plt.plot(sd_ai, label="AI", color="tab:blue")
    plt.plot(sd_h, label="Human", color="tab:orange")
    plt.xlabel("Iteracja")
    plt.ylabel("SD (polaryzacja)")
    plt.title("Polaryzacja (SD) w czasie")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # --- ROZKŁAD KOŃCOWY ---
    bins = np.linspace(-1, 1, 40)
    plt.figure(figsize=(7,4))
    plt.hist(end_ai, bins=bins, density=True, alpha=0.6,
             label="AI", color="tab:blue")
    plt.hist(end_h, bins=bins, density=True, alpha=0.6,
             label="Human", color="tab:orange")
    plt.xlabel("Opinia")
    plt.ylabel("Gęstość")
    plt.title("Rozkład opinii po T iteracjach")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_summary_bars(ai, human):
    start_ai, end_ai, mean_ai, sd_ai = ai
    start_h,  end_h,  mean_h,  sd_h  = human

    # start jest identyczny – bierzemy AI jako referencję
    start = start_ai

    labels = ["Start", "Koniec AI", "Koniec Human"]

    # ===== ŚREDNIE =====
    means = [
        start.mean(),
        end_ai.mean(),
        end_h.mean()
    ]

    # ===== SD (surowe) =====
    sd_start = start.std(ddof=1)
    sd_ai_end = end_ai.std(ddof=1)
    sd_h_end  = end_h.std(ddof=1)

    # ===== SD przeskalowane względem startu =====
    sds_scaled = [
        1.0,
        sd_ai_end / sd_start,
        sd_h_end  / sd_start
    ]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # -----------------------
    # Panel 1: średnia
    # -----------------------
    ax[0].bar(labels, means)
    ax[0].axhline(0, color="grey", linestyle="--", alpha=0.6)
    ax[0].set_ylim(-1, 1)
    ax[0].set_title("Średnia opinii (Start vs Koniec)")
    ax[0].set_ylabel("Średnia opinia (-1..+1)")
    ax[0].grid(axis="y", alpha=0.25)

    # -----------------------
    # Panel 2: polaryzacja (SD wzgl. startu)
    # -----------------------
    ax[1].bar(labels, sds_scaled)
    ax[1].axhline(1.0, color="grey", linestyle="--", alpha=0.6)
    ax[1].set_title("Zmiana polaryzacji względem startu")
    ax[1].set_ylabel("SD / SD_start")
    ax[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_histograms_start_ai_human(start, end_ai, end_h):
    bins = np.linspace(-1, 1, 40)

    plt.figure(figsize=(7, 4))

    # START
    plt.hist(
        start,
        bins=bins,
        density=True,
        alpha=0.35,
        label="Start",
        color="grey"
    )

    # AI
    plt.hist(
        end_ai,
        bins=bins,
        density=True,
        alpha=0.6,
        label="AI",
        color="tab:blue"
    )

    # HUMAN
    plt.hist(
        end_h,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Human",
        color="tab:orange"
    )

    plt.xlabel("Opinia (-1 … +1)")
    plt.ylabel("Gęstość")
    plt.title("Rozkład opinii: Start vs AI vs Human")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

#=== Wykresy do repliakcji
def plot_mean_with_uncertainty(rep_ai, rep_h):
    m_ai = rep_ai["mean"].mean(axis=0)
    m_h  = rep_h["mean"].mean(axis=0)

    lo_ai, hi_ai = np.percentile(rep_ai["mean"], [10, 90], axis=0)
    lo_h,  hi_h  = np.percentile(rep_h["mean"],  [10, 90], axis=0)

    plt.figure(figsize=(7,4))

    plt.plot(m_ai, label="AI", color="tab:blue")
    plt.fill_between(range(len(m_ai)), lo_ai, hi_ai,
                     color="tab:blue", alpha=0.2)

    plt.plot(m_h, label="Human", color="tab:orange")
    plt.fill_between(range(len(m_h)), lo_h, hi_h,
                     color="tab:orange", alpha=0.2)

    plt.axhline(0, color="grey", linestyle="--", alpha=0.5)
    plt.xlabel("Iteracja")
    plt.ylabel("Średnia opinia")
    plt.title("Średnia opinia (replikacje)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_sd_with_uncertainty(rep_ai, rep_h):
    sd_ai = rep_ai["sd"]
    sd_h  = rep_h["sd"]

    m_ai = sd_ai.mean(axis=0)
    m_h  = sd_h.mean(axis=0)

    lo_ai, hi_ai = np.percentile(sd_ai, [10, 90], axis=0)
    lo_h,  hi_h  = np.percentile(sd_h,  [10, 90], axis=0)

    plt.figure(figsize=(7,4))

    plt.plot(m_ai, label="AI", color="tab:blue")
    plt.fill_between(range(len(m_ai)), lo_ai, hi_ai,
                     color="tab:blue", alpha=0.2)

    plt.plot(m_h, label="Human", color="tab:orange")
    plt.fill_between(range(len(m_h)), lo_h, hi_h,
                     color="tab:orange", alpha=0.2)

    plt.xlabel("Iteracja")
    plt.ylabel("SD (polaryzacja)")
    plt.title("Polaryzacja (replikacje)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    ai = run_simulation(source="AI", seed=123)
    human = run_simulation(source="HUMAN", seed=123)

    start_ai, end_ai, _, _ = ai
    start_h,  end_h,  _, _ = human

    start = start_ai

    plot_compare(ai, human)
    plot_summary_bars(ai, human)
    plot_histograms_start_ai_human(start_ai, end_ai, end_h)

    rep_ai = replicate_simulation("AI", R=30, N=500, T=16, seed=123)
    rep_h  = replicate_simulation("HUMAN", R=30, N=500, T=16, seed=123)

    plot_mean_with_uncertainty(rep_ai, rep_h)
    plot_sd_with_uncertainty(rep_ai, rep_h)




