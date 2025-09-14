# app.py
import os
import time
import traceback
from flask import Flask, render_template, request, redirect, url_for, flash
import subprocess

# Import the functions from your dsp_full_flow.py
# Make sure dsp_full_flow.py is in same folder and exposes these names.
from full_flow import (
    ensure_dirs,
    run_sine,
    design_fir,
    design_firwin,
    save_coeffs_q5_27,
    plot_filter_response,
    run_verilog_sim_with_bash_function,
    analyze_verilog_output,
    run_ideal_response,
    compare_ideal_vs_verilog,
)

app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret"  # change for production

# paths used for templates/static
PLOTS_DIR = os.path.join("static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
ensure_dirs()

@app.route("/", methods=["GET"])
def index():
    # default parameters shown on form
    defaults = {
        "fs": 8000,
        "filter_type": "lowpass",
        "ripple": 1,
        "attenuation": 30,
        "f_pass": 1000,
        "f_stop": 1200,
        "f_stop1": 400,
        "f_pass1": 600,
        "f_pass2": 1400,
        "f_stop2": 1600,
        "sine_freqs": "1000",
        "numtaps": "63"
    }
    return render_template("index.html", defaults=defaults)

@app.route("/run", methods=["POST"])
def run_flow():
    try:
        fs = float(request.form.get("fs", "8000"))
        method = request.form.get("method", "remez")
        ftype = request.form.get("filter_type", "lowpass")
        ripple = float(request.form.get("ripple", "0.5"))
        attenuation = float(request.form.get("attenuation", "60"))
        numtaps_raw = request.form.get("numtaps", "").strip()
        numtaps = int(numtaps_raw) if numtaps_raw else None

        # === Parse filter params ===
        if method == "remez":
            if ftype == "lowpass":
                edges = [float(request.form.get("f_pass")), float(request.form.get("f_stop"))]
            elif ftype == "highpass":
                edges = [float(request.form.get("f_stop")), float(request.form.get("f_pass"))]
            elif ftype == "bandpass":
                edges = [
                    float(request.form.get("f_stop1")),
                    float(request.form.get("f_pass1")),
                    float(request.form.get("f_pass2")),
                    float(request.form.get("f_stop2")),
                ]
            else:  # bandstop
                edges = [
                    float(request.form.get("f_pass1")),
                    float(request.form.get("f_stop1")),
                    float(request.form.get("f_stop2")),
                    float(request.form.get("f_pass2")),
                ]
        else:  # FIRWIN
            cutoff_str = request.form.get("cutoff", "").strip()
            if cutoff_str.startswith("["):
                cutoff = eval(cutoff_str)   # e.g. [0.2,0.5]
            elif "," in cutoff_str:
                cutoff = [float(x) for x in cutoff_str.split(",") if x.strip()]
            else:
                cutoff = float(cutoff_str)

            window = request.form.get("window", "hamming")
            beta_raw = request.form.get("beta", "").strip()
            beta = float(beta_raw) if beta_raw else None
            pass_zero = request.form.get("pass_zero", "true") == "true"

        # Parse sine freqs
        sine_freqs = [
            float(x.strip())
            for x in request.form.get("sine_freqs", "1000").split(",")
            if x.strip()
        ]

        os.makedirs("plots", exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # === Run flow ===
        t, wave = run_sine(fs, sine_freqs, duration=1.0)

        if method == "remez":
            taps = design_fir(fs, ripple, attenuation, ftype, edges, numtaps)
            plot_filter_response(fs, taps, ftype, edges, ripple, attenuation, numtaps)
        else:
            if numtaps is None:
                numtaps = 63  # sensible default
            taps = design_firwin(fs, numtaps, cutoff, window, pass_zero, beta)
            # reuse plot_filter_response for consistency
            plot_filter_response(fs, taps, ftype, [], ripple, attenuation, numtaps)

        save_coeffs_q5_27(taps, path="output/coeff.txt")
        subprocess.run(f"cp plots/filter_response.png {PLOTS_DIR}/filter_response.png", shell=True)

        run_verilog_sim_with_bash_function()
        verilog_y = analyze_verilog_output(fs, in_path="input/filter_response_verilog.txt")

        for src in ("filter_out_fft_verilog.png", "filter_out_time_verilog.png"):
            if os.path.exists(f"plots/{src}"):
                subprocess.run(f"cp plots/{src} {PLOTS_DIR}/{src}", shell=True)

        ideal_y = run_ideal_response(fs, taps, sine_freqs, duration=1.0)
        for src in ("ideal_response_time.png", "ideal_response_fft.png"):
            if os.path.exists(f"plots/{src}"):
                subprocess.run(f"cp plots/{src} {PLOTS_DIR}/{src}", shell=True)

        rmse = compare_ideal_vs_verilog(fs)
        for src in ("comparison_time.png", "comparison_fft.png"):
            if os.path.exists(f"plots/{src}"):
                subprocess.run(f"cp plots/{src} {PLOTS_DIR}/{src}", shell=True)

        results = {
            "method": method,
            "fs": fs,
            "ftype": ftype,
            "numtaps": len(taps),
            "sine_freqs": sine_freqs,
            "rmse": float(rmse),
        }

        if method == "remez":
            results.update({
                "ripple": ripple,
                "attenuation": attenuation,
                "edges": edges,
            })
        else:  # FIRWIN
            results.update({
                "cutoff": cutoff,
                "window": window,
                "pass_zero": pass_zero,
                "beta": beta,
            })
        return render_template("results.html", results=results, plots_dir=f"/static/plots")
    except Exception as e:
        trace = traceback.format_exc()
        flash(f"Error during full flow: {e}\nSee server logs for details.")
        print(trace)
        return redirect(url_for("index"))

if __name__ == "__main__":
    # For development only: set debug=False when running on shared machine
    app.run(host="0.0.0.0", port=5000, debug=True)

