# dsp_full_flow.py
# One-file pipeline: Sine gen → FIR design → ModelSim via your bash function → Verilog analysis → Ideal response → Comparison
# Requires: numpy, matplotlib, scipy (remez, lfilter)

import os
import sys
import shlex
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import remez, lfilter, freqz

# ---------------------------
# Utility: Q formats & I/O
# ---------------------------

def float_to_q1_31_hex(value: float) -> str:
    """Convert float (-1 to <1) to signed Q1.31 hex (8 hex chars)."""
    scaled = int(np.round(value * (2**31)))
    if scaled < -2**31: scaled = -2**31
    if scaled >  2**31 - 1: scaled =  2**31 - 1
    return f"{scaled & 0xFFFFFFFF:08X}"

def float_to_q5_27_hex(value: float) -> str:
    """Convert float (~-16 to +16 range) to signed Q5.27 hex (8 hex chars)."""
    scaled = int(np.round(value * (2**27)))
    if scaled < -2**31: scaled = -2**31
    if scaled >  2**31 - 1: scaled =  2**31 - 1
    return f"{scaled & 0xFFFFFFFF:08X}"

def q5_27_hex_to_float(hexstr: str) -> float:
    """Convert Q5.27 hex (8 chars) to float."""
    val = int(hexstr, 16)
    if val & 0x80000000:
        val = -((~val & 0xFFFFFFFF) + 1)
    return val / (2**27)

def ensure_dirs():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

# ---------------------------
# Sine generation (non-interactive)
# ---------------------------

def run_sine(fs: float, freqs: list[float], duration: float = 1.0):
    """Generate combined sine, normalize, write Q1.31 to output/sine_wave_quant.txt, and plot."""
    ensure_dirs()
    num_samples = int(fs * duration)
    t = np.arange(num_samples) / fs
    wave = np.zeros(num_samples, dtype=np.float64)
    for f in freqs:
        wave += np.sin(2 * np.pi * f * t)
    max_abs = np.max(np.abs(wave))
    if max_abs > 0:
        wave /= max_abs

    # Save Q1.31 input for Verilog
    with open("output/sine_wave_quant.txt", "w") as f:
        for v in wave:
            f.write(float_to_q1_31_hex(v) + "\n")

    # Quick zoom plot: ~5 cycles of lowest frequency
    f_min = max(min(freqs) if freqs else 1.0, 1e-9)
    samples_to_show = int(min(num_samples, (5 * fs / f_min)))
    plt.figure(figsize=(10, 5))
    plt.plot(t[:samples_to_show], wave[:samples_to_show])
    plt.title(f"Sine Input (float) — zoomed ~5 cycles of {f_min} Hz")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid(True)
    plt.tight_layout(); plt.savefig("plots/sine_input_plot_zoomed.png")

    print(f"✅ Sine generated: fs={fs}, freqs={freqs}, samples={num_samples}")
    return t, wave

# ---------------------------
# FIR design (remez) for all types
# ---------------------------

def _db_to_linear(ripple_db: float, attenuation_db: float):
    delta_p = 10**(ripple_db/20.0) - 1.0
    delta_s = 10**(-attenuation_db/20.0)
    return delta_p, delta_s

def design_fir(fs: float, ripple_db: float, attenuation_db: float,
               filter_type: str, edges: list[float], numtaps: int | None):
    """
    filter_type: 'lowpass','highpass','bandpass','bandstop'
    edges:
      LP : [f_pass, f_stop]
      HP : [f_stop, f_pass]
      BP : [f_stop1, f_pass1, f_pass2, f_stop2]
      BS : [f_pass1, f_stop1, f_stop2, f_pass2]
    """
    delta_p, delta_s = _db_to_linear(ripple_db, attenuation_db)

    if filter_type == "lowpass":
        f_pass, f_stop = edges
        bands = [0, f_pass, f_stop, fs/2]
        desired = [1, 0]
        weight = [1, delta_p/delta_s]
    elif filter_type == "highpass":
        f_stop, f_pass = edges
        bands = [0, f_stop, f_pass, fs/2]
        desired = [0, 1]
        weight = [delta_p/delta_s, 1]
    elif filter_type == "bandpass":
        f_stop1, f_pass1, f_pass2, f_stop2 = edges
        bands = [0, f_stop1, f_pass1, f_pass2, f_stop2, fs/2]
        desired = [0, 1, 0]
        weight  = [delta_p/delta_s, 1, delta_p/delta_s]
    elif filter_type == "bandstop":
        f_pass1, f_stop1, f_stop2, f_pass2 = edges
        bands = [0, f_pass1, f_stop1, f_stop2, f_pass2, fs/2]
        desired = [1, 0, 1]
        weight  = [1, delta_p/delta_s, 1]
    else:
        raise ValueError("Invalid filter_type")

    # Estimate taps if needed (Kaiser-like estimate)
    if numtaps is None:
        # use edge transition width as min |adjacent-band-edge diffs|
        bw_candidates = []
        for i in range(1, len(bands)):
            bw_candidates.append(abs(bands[i] - bands[i-1]))
        delta_f = min(bw_candidates) / fs
        A = max(ripple_db, attenuation_db)
        est = int((A - 8) / (2.285 * max(delta_f, 1e-6))) + 1
        if est % 2 == 0:
            est += 1
        numtaps = max(est, 11)
        print(f"Estimated taps: {numtaps}")
    else:
        print(f"Using taps: {numtaps}")
    print(bands)
    taps = remez(numtaps, bands, desired, weight=weight, fs=fs)
    return taps

def save_coeffs_q5_27(taps: np.ndarray, path: str = "output/coeff.txt"):
    ensure_dirs()
    with open(path, "w") as f:
        for c in taps:
            f.write(float_to_q5_27_hex(float(c)) + "\n")
    print(f"✅ Quantized coeffs (Q5.27) saved to {path}")


def plot_filter_response(fs, taps, filter_type, edges, ripple, attenuation, numtaps):
    ensure_dirs()
    w, h = freqz(taps, worN=2048, fs=fs)
    plt.figure(figsize=(10,5))
    plt.plot(w, 20*np.log10(np.maximum(np.abs(h), 1e-10)))
    
    # draw vertical lines at band edges
    for f in edges:
        plt.axvline(f, color="r", linestyle="--")
    # passband ripple lines
    plt.axhline(ripple, color="green", linestyle="--", linewidth=1.2, label="Passband ripple")
    plt.axhline(-ripple, color="green", linestyle="--", linewidth=1.2)
    # stopband attenuation line
    plt.axhline(-attenuation, color="red", linestyle="--", linewidth=1.2, label="Stopband attenuation")
    # -3 dB ref line
    plt.axhline(-3, color="blue", linestyle="--", linewidth=1.2, label="-3 dB")
    
    plt.title(f"{filter_type.upper()} Filter — {len(taps)} Taps")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude (dB)")
    plt.grid(True); plt.tight_layout()
    plt.savefig("plots/filter_response.png")
    plt.close()
    print("✅ Filter design plot saved → plots/filter_response.png")

# ---------------------------
# Run Verilog simulation using your bash function
# ---------------------------

def run_verilog_sim_with_bash_function():
    """
    Calls your interactive bash shell function:
        simulate tb_fir_filter FIR_filter.v TB_FIR_filter.v
    """
    #subprocess.run("bash -i -c 'simulate tb_fir_filter /home/harsh/Projects/FIR/verilog/FIR_filter/FIR_filter.v /home/harsh/Projects/FIR/verilog/FIR_filter/TB_FIR_filter.v'",

    cmd = "bash -i -c 'simulate_headless tb_fir_filter /home/harsh/Projects/FIR_gui/verilog/FIR_filter/FIR_filter.v /home/harsh/Projects/FIR_gui/verilog/FIR_filter/TB_FIR_filter.v'"
    print(f"\n→ {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    print("✅ Verilog simulation finished.")

# ---------------------------
# Analyze Verilog output (Q5.27 hex -> float) and plot
# ---------------------------

def analyze_verilog_output(fs: float,
                           in_path: str = "input/filter_response_verilog.txt"):
    ensure_dirs()
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Verilog output not found: {in_path}")

    vals = []
    with open(in_path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                vals.append(q5_27_hex_to_float(s))
    y = np.array(vals, dtype=np.float64)

    # FFT
    N = len(y)
    freqs = np.fft.fftfreq(N, d=1/fs)
    Y = np.abs(np.fft.fft(y)) / N
    half = N // 2
    dom_idx = np.argmax(Y[:half])
    dom_freq = freqs[:half][dom_idx]

    # Plot spectrum
    plt.figure(figsize=(12, 5))
    plt.plot(freqs[:half], Y[:half], linewidth=1)
    plt.title("Verilog Output — Frequency Spectrum")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude"); plt.grid(True)
    plt.tight_layout(); plt.savefig("plots/filter_out_fft_verilog.png")

    # Plot time (auto ~10 cycles if dom_freq>0)
    t = np.arange(N) / fs
    if dom_freq > 0:
        samples_to_show = int(min(N, fs * (10.0 / dom_freq)))
    else:
        samples_to_show = min(N, 200)
    plt.figure(figsize=(12, 4))
    plt.plot(t[:samples_to_show], y[:samples_to_show], linewidth=1.2)
    plt.title(f"Verilog Output — Time Domain (showing {samples_to_show} samples)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid(True)
    plt.tight_layout(); plt.savefig("plots/filter_out_time_verilog.png")

    print(f"🔎 Verilog dominant frequency ≈ {dom_freq:.2f} Hz")
    return y

# ---------------------------
# Ideal response (apply FIR to float sine) and plot
# ---------------------------

def run_ideal_response(fs: float, taps: np.ndarray, sine_freqs: list[float], duration: float = 1.0):
    ensure_dirs()
    # regenerate same sine as float (to match what got quantized for Verilog input)
    num_samples = int(fs * duration)
    t = np.arange(num_samples) / fs
    x = np.zeros(num_samples, dtype=np.float64)
    for f in sine_freqs:
        x += np.sin(2 * np.pi * f * t)
    if np.max(np.abs(x)) > 0:
        x /= np.max(np.abs(x))

    y = lfilter(taps, 1.0, x)
    np.save("output/ideal_response.npy", y)

    # Time plot (first 500 samples)
    plt.figure(figsize=(12, 5))
    plt.plot(t[:500], y[:500])
    plt.title("Ideal Filtered Output — Time Domain (First 500 samples)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid(True)
    plt.tight_layout(); plt.savefig("plots/ideal_response_time.png")

    # FFT plot
    N = len(y)
    freqs = np.fft.fftfreq(N, 1/fs)
    Y = np.abs(np.fft.fft(y)) / N
    plt.figure(figsize=(12, 5))
    plt.plot(freqs[:N//2], Y[:N//2])
    plt.title("Ideal Filtered Output — Frequency Spectrum")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude"); plt.grid(True)
    plt.tight_layout(); plt.savefig("plots/ideal_response_fft.png")

    print("✅ Ideal response saved: plots + output/ideal_response.npy")
    return y

# ---------------------------
# Comparison (time/FFT + RMSE)
# ---------------------------

def compare_ideal_vs_verilog(fs: float,
                             ideal_path: str = "output/ideal_response.npy",
                             verilog_path: str = "input/filter_response_verilog.txt"):
    if not os.path.exists(ideal_path):
        raise FileNotFoundError("Ideal response not found. Run ideal step first.")
    ideal = np.load(ideal_path)

    if not os.path.exists(verilog_path):
        raise FileNotFoundError("Verilog response not found.")
    verilog = []
    with open(verilog_path, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                verilog.append(q5_27_hex_to_float(s))
    verilog = np.array(verilog, dtype=np.float64)

    N = min(len(ideal), len(verilog))
    ideal = ideal[:N]
    verilog = verilog[:N]

    rmse = float(np.sqrt(np.mean((ideal - verilog)**2)))
    print(f"📏 RMSE (Ideal vs Verilog): {rmse:.6e}")

    # Time overlay
    plt.figure(figsize=(12, 5))
    plt.plot(ideal[:500], label="Ideal (Python)")
    plt.plot(verilog[:500], label="Verilog")
    plt.title("Time-Domain Comparison (First 500 samples)")
    plt.xlabel("Sample"); plt.ylabel("Amplitude"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("plots/comparison_time.png")

    # FFT overlay
    F = np.fft.fftfreq(N, 1/fs)
    FFT_i = np.abs(np.fft.fft(ideal))/N
    FFT_v = np.abs(np.fft.fft(verilog))/N
    half = N//2
    plt.figure(figsize=(12, 5))
    plt.plot(F[:half], FFT_i[:half], label="Ideal FFT")
    plt.plot(F[:half], FFT_v[:half], label="Verilog FFT")
    plt.title("Frequency-Domain Comparison")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig("plots/comparison_fft.png")

    print("✅ Comparison saved: plots/comparison_time.png, plots/comparison_fft.png")
    return rmse

# ---------------------------
# Full Flow (single prompt set)
# ---------------------------

def prompt_params_once():
    print("\n=== DSP Full Flow Parameters ===")
    fs = float(input("Sampling frequency fs (Hz): ").strip())

    print("\nFilter type options: 1=lowpass, 2=highpass, 3=bandpass, 4=bandstop")
    ft_choice = input("Choose filter type [1/2/3/4]: ").strip()
    type_map = {"1":"lowpass","2":"highpass","3":"bandpass","4":"bandstop"}
    ftype = type_map.get(ft_choice, "lowpass")

    ripple = float(input("Passband ripple (dB): ").strip())
    attenuation = float(input("Stopband attenuation (dB): ").strip())

    if ftype == "lowpass":
        f_pass = float(input("Passband edge f_pass (Hz): ").strip())
        f_stop = float(input("Stopband edge f_stop (Hz): ").strip())
        edges = [f_pass, f_stop]
    elif ftype == "highpass":
        f_stop = float(input("Stopband edge f_stop (Hz): ").strip())
        f_pass = float(input("Passband edge f_pass (Hz): ").strip())
        edges = [f_stop, f_pass]
    elif ftype == "bandpass":
        f_stop1 = float(input("Lower stopband edge f_stop1 (Hz): ").strip())
        f_pass1 = float(input("Lower passband edge f_pass1 (Hz): ").strip())
        f_pass2 = float(input("Upper passband edge f_pass2 (Hz): ").strip())
        f_stop2 = float(input("Upper stopband edge f_stop2 (Hz): ").strip())
        edges = [f_stop1, f_pass1, f_pass2, f_stop2]
    else:  # bandstop
        f_pass1 = float(input("Lower passband edge f_pass1 (Hz): ").strip())
        f_stop1 = float(input("Lower stopband edge f_stop1 (Hz): ").strip())
        f_stop2 = float(input("Upper stopband edge f_stop2 (Hz): ").strip())
        f_pass2 = float(input("Upper passband edge f_pass2 (Hz): ").strip())
        edges = [f_pass1, f_stop1, f_stop2, f_pass2]

    sine_freqs = [float(x) for x in input("Sine freqs (comma-separated, Hz): ").split(",") if x.strip()]
    taps_in = input("Number of taps (Enter for auto): ").strip()
    numtaps = int(taps_in) if taps_in else None

    return fs, ftype, ripple, attenuation, edges, numtaps, sine_freqs

def full_flow():
    ensure_dirs()
    fs, ftype, ripple, attenuation, edges, numtaps, sine_freqs = prompt_params_once()

    # 1) Sine input (Q1.31 for Verilog)
    run_sine(fs, sine_freqs, duration=1.0)

    # 2) FIR design (remez) → save coeffs Q5.27 for Verilog
    taps = design_fir(fs, ripple, attenuation, ftype, edges, numtaps)
    save_coeffs_q5_27(taps, "output/coeff.txt")
    plot_filter_response(fs, taps, ftype, edges, ripple, attenuation, numtaps)
    # 3) Run your Verilog simulation via bash function
    run_verilog_sim_with_bash_function()

    # 4) Analyze Verilog output (time + FFT)
    verilog_y = analyze_verilog_output(fs, "input/filter_response_verilog.txt")

    # 5) Ideal Python response (time + FFT), save samples
    ideal_y = run_ideal_response(fs, taps, sine_freqs, duration=1.0)

    # 6) Compare + RMSE
    rmse = compare_ideal_vs_verilog(fs)

    print("\n=== DONE ===")
    print(f"Filter type: {ftype}, fs={fs} Hz, ripple={ripple} dB, stop attn={attenuation} dB")
    print(f"Edges: {edges}, taps={len(taps)}; Sine freqs: {sine_freqs}")
    print(f"RMSE vs Verilog: {rmse:.6e}")
    print("Artifacts:")
    print("  - Input: output/sine_wave_quant.txt")
    print("  - Coeffs: output/coeff.txt")
    print("  - Verilog plots: plots/filter_out_time_verilog.png, plots/filter_out_fft_verilog.png")
    print("  - Ideal plots: plots/ideal_response_time.png, plots/ideal_response_fft.png")
    print("  - Comparison: plots/comparison_time.png, plots/comparison_fft.png")

# ---------------------------
# Entry
# ---------------------------

if __name__ == "__main__":
    try:
        full_flow()
    except subprocess.CalledProcessError as e:
            print("\n❌ External command failed.")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

