import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Wczytanie danych
df = pd.read_csv("GR2A.csv", sep=';', decimal=',')

x = df.iloc[:, 0] / 25  # Przeliczenie próbek na sekundy
y1 = df.iloc[:, 1]
y2 = df.iloc[:, 2]


# Projektowanie filtru Butterwortha
def butter_lowpass_filter(data, cutoff=6, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# Filtracja
y1_filtered = butter_lowpass_filter(y1, cutoff=6, fs=100)
y2_filtered = butter_lowpass_filter(y2, cutoff=6, fs=100)

# Filtracja filtrem o ruchomej średniej
y1_smooth = pd.Series(y1_filtered).rolling(window=5, center=True).mean()
y2_smooth = pd.Series(y2_filtered).rolling(window=5, center=True).mean()

# Przedziały czasowe (w sekundach)
segments = {
    "Linia podstawowa": (20, 25),
    "Ćwiczenie 1 (podest)": [(70, 100), (120, 130)],
    "Ćwiczenie 2 (paluszki)": (180, 205),
    "Ćwiczenie 3 (balerinki)": (215, 245),
    "Ćwiczenie 4 (przysiad - artefakt)": (265, 295)
}


# Funkcja do rysowania surowego i obrobionego sygnału
def plot_segment(title, time_range, x, y1_raw, y2_raw, y1_filt, y2_filt):
    plt.figure(figsize=(12, 5))

    # Wybranie tylko danych w zakresie czasowym
    if isinstance(time_range, list):  # Jeśli segment ma kilka zakresów (Ćwiczenie 1)
        for i, tr in enumerate(time_range):
            mask = (x >= tr[0]) & (x <= tr[1])
            plt.plot(x[mask], y1_raw[mask], color="red", alpha=0.3, label="Głowa boczna (surowy)" if i == 0 else "")
            plt.plot(x[mask], y2_raw[mask], color="green", alpha=0.3,
                     label="Głowa pośrodkowa (surowy)" if i == 0 else "")

    else:
        mask = (x >= time_range[0]) & (x <= time_range[1])
        plt.plot(x[mask], y1_raw[mask], color="red", alpha=0.3, label="Głowa boczna (surowy)")
        plt.plot(x[mask], y2_raw[mask], color="green", alpha=0.3, label="Głowa pośrodkowa (surowy)")

    plt.xlabel("Czas [s]")
    plt.ylabel("Napięcie [mV]")
    plt.title(f"{title} - PRZED filtracją")
    plt.legend()
    plt.grid()
    plt.show()

    # Wykres po filtracji
    plt.figure(figsize=(12, 5))

    if isinstance(time_range, list):
        for i, tr in enumerate(time_range):
            mask = (x >= tr[0]) & (x <= tr[1])
            plt.plot(x[mask], y1_filt[mask], color="red", label="Głowa boczna (obrobiony)" if i == 0 else "")
            plt.plot(x[mask], y2_filt[mask], color="green", label="Głowa pośrodkowa (obrobiony)" if i == 0 else "")

    else:
        mask = (x >= time_range[0]) & (x <= time_range[1])
        plt.plot(x[mask], y1_filt[mask], color="red", label="Głowa boczna (obrobiony)")
        plt.plot(x[mask], y2_filt[mask], color="green", label="Głowa pośrodkowa (obrobiony)")

    plt.xlabel("Czas [s]")
    plt.ylabel("Napięcie [mV]")
    plt.title(f"{title} - PO filtracji")
    plt.legend()
    plt.grid()
    plt.show()


# Rysowanie wykresów dla każdego przedziału czasowego (surowy i obrobiony)
for name, time_range in segments.items():
    plot_segment(name, time_range, x, y1, y2, y1_smooth, y2_smooth)
