import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. KOKPIT STEROWNICZY
# ==========================================

# A. PARAMETRY POPULACJI
NUM_AGENTS = 1000          # Liczba agentów
ITERATIONS = 40            # Jak długo trwa eksperyment (czas)

# B. PARAMETRY PSYCHOLOGICZNE

# ŹRÓDŁO: Sherif & Hovland (1961) - Social Judgment Theory.
# Hipoteza: Osoby silnie spolaryzowane mają węższy zakres akceptacji.
# WARTOŚĆ: Testuj zakres 0.3 (radykałowie) do 0.7 (liberałowie/otwarci).
LATITUDE_ACCEPTANCE = 0.4  

# ŹRÓDŁO: Pilotaż (Pytanie o irytację/odrzucenie treści).
# Jeśli osoba o poglądzie 0.0 odrzuca treść 0.7 jako "bzdurę", to Lat_Rejection = 0.7.
# Mniejsza liczba = łatwiej kogoś zradykalizować (szybciej "pęka").
LATITUDE_REJECTION = 0.7   

# ŹRÓDŁO: Teoria Reaktancji (Brehm, 1966).
# Jak mocno "odbijamy się" od ściany, gdy ktoś nas zmusza do zmiany zdania?
# 0.1 = lekki opór, 1.0 = robimy dokładnie na złość (pełna polaryzacja).
BOOMERANG_STRENGTH = 0.5   

# C. PARAMETRY ZAUFANIA (Główna hipoteza doktoratu)

# ŹRÓDŁO: Pilotaż -> Pytanie "Na ile ten tekst jest wiarygodny?" (Skala 0-1)
# Średnia ocena dla tekstów oznaczonych etykietą "Wygenerowane przez AI".
# Literatura (Sundar, 2020) sugeruje Machine Heuristic -> wyższe zaufanie do obiektywizmu (ok. 0.7-0.8).
TRUST_AI_MEAN = 0.8        

# ŹRÓDŁO: Pilotaż -> Średnia ocena wiarygodności dla tekstów "Autor: Jan Kowalski".
# Zazwyczaj niższa niż AI dla faktów (błędy poznawcze), ale wyższa dla opinii.
TRUST_HUMAN_MEAN = 0.5     

# D. BODZIEC (Co im pokazujemy?)
# 1.0 = Skrajna Prawica/Opcja B, -1.0 = Skrajna Lewica/Opcja A
RECOMMENDATION_VALUE = 0.3
SOURCE_TYPE = 'AI'         # 'AI' lub 'HUMAN'

# ==========================================
# 2. SILNIK SYMULACJI
# ==========================================

class Agent:
    def __init__(self, opinion, susceptibility, trust_ai, trust_human):
        self.opinion = opinion
        self.susceptibility = susceptibility
        self.trust_ai = trust_ai
        self.trust_human = trust_human
        
        # Przypisanie parametrów z konfiguracji
        self.lat_acceptance = LATITUDE_ACCEPTANCE
        self.lat_rejection = LATITUDE_REJECTION

    def update_opinion(self, source_type, recommendation):
        # Ustalenie wagi zaufania w zależności od źródła
        if source_type == 'AI':
            trust = self.trust_ai
        else:
            trust = self.trust_human

        # Podstawowa siła wpływu (zaufanie * podatność osobista)
        influence = trust * self.susceptibility
        
        # Oblicz dystans: Jak daleko jest treść od mojej głowy?
        dist = recommendation - self.opinion
        
        # LOGIKA POLARYZACJI (Sherif & Hovland)
        
        # 1. Strefa Akceptacji (Asymilacja) -> Zgadzam się i przybliżam
        if abs(dist) < self.lat_acceptance:
            change = influence * dist
            
        # 2. Strefa Odrzucenia (Kontrast) -> Wkurzam się i oddalam (Efekt Bumerangowy)
        elif abs(dist) > self.lat_rejection:
            # Znak przeciwny do dystansu (ucieczka w drugą stronę)
            # Mnożymy przez siłę bumerangu
            change = - (influence * BOOMERANG_STRENGTH) * np.sign(dist)
            
        # 3. Strefa Obojętności -> Nic się nie dzieje
        else:
            change = 0

        # Aktualizacja opinii i pilnowanie zakresu [-1, 1]
        self.opinion += change
        self.opinion = max(-1, min(1, self.opinion))

# --- INICJALIZACJA ---
print(f"Start symulacji: Źródło={SOURCE_TYPE}, Rekomendacja={RECOMMENDATION_VALUE}")

# Generujemy populację o rozkładzie normalnym (większość w centrum)
agents = []
for _ in range(NUM_AGENTS):
    # Opinie: Rozkład normalny (średnia 0, odchylenie 0.4)
    op = np.random.normal(0, 0.4)
    # Clip do [-1, 1]
    op = max(-1, min(1, op))
    
    # Zaufanie i podatność (losowe w pewnym zakresie wokół średniej z konfiguracji)
    susp = np.random.uniform(0.1, 0.9)
    tr_ai = np.random.normal(TRUST_AI_MEAN, 0.1)
    tr_hu = np.random.normal(TRUST_HUMAN_MEAN, 0.1)
    
    agents.append(Agent(op, susp, tr_ai, tr_hu))

# Zapisujemy stan "PRZED"
opinions_start = [a.opinion for a in agents]

# --- PĘTLA CZASU ---
history = []
for i in range(ITERATIONS):
    current_avg = np.mean([a.opinion for a in agents])
    history.append(current_avg)
    
    # Każdy agent spotyka się z bodźcem
    for agent in agents:
        agent.update_opinion(SOURCE_TYPE, RECOMMENDATION_VALUE)

# Zapisujemy stan "PO"
opinions_end = [a.opinion for a in agents]

# ==========================================
# 3. WIZUALIZACJA WYNIKÓW
# ==========================================
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Wykres A: Średnia opinia (czy tłum uległ?)
ax[0].plot(history, linewidth=3, color='#3498db')
ax[0].set_title(f"Dynamika Średniej Opinii (Źródło: {SOURCE_TYPE})", fontsize=14)
ax[0].set_xlabel("Czas (iteracje)")
ax[0].set_ylabel("Opinia (-1 Lewica ... +1 Prawica)")
ax[0].set_ylim(-1, 1.1)
ax[0].grid(True, alpha=0.3)
ax[0].axhline(RECOMMENDATION_VALUE, color='green', linestyle='--', label='Pozycja Agenta')
ax[0].legend()

# Wykres B: Rozkład (GĘSTOŚĆ) - Tu widać polaryzację
sns.kdeplot(opinions_start, ax=ax[1], fill=True, color='grey', alpha=0.3, label='Początek (T=0)')
sns.kdeplot(opinions_end, ax=ax[1], fill=True, color='red', alpha=0.5, label=f'Koniec (T={ITERATIONS})')

ax[1].set_title("Zmiana Rozkładu Społecznego (Polaryzacja)", fontsize=14)
ax[1].set_xlabel("Spektrum Opinii")
ax[1].set_xlim(-1.1, 1.1)
ax[1].axvline(RECOMMENDATION_VALUE, color='green', linestyle='--', label='Stanowisko Agenta')

# Zaznaczamy strefy (dla celów edukacyjnych na wykresie)
if SOURCE_TYPE == 'AI':
    plt.text(-0.8, 0.5, f"Latitude Rejection: {LATITUDE_REJECTION}", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

ax[1].legend()

plt.tight_layout()
plt.savefig('/workspaces/doktorat/plot.png', dpi=150, bbox_inches='tight')
print("✓ Wykres zapisany jako plot.png")