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
    """Run the full flow synchronously and redirect to results."""
    try:
        # 1. Read form params (one-shot)
        fs = float(request.form.get("fs", "8000"))
        ftype = request.form.get("filter_type", "lowpass")
        ripple = float(request.form.get("ripple", "0.5"))
        attenuation = float(request.form.get("attenuation", "60"))
        numtaps_raw = request.form.get("numtaps", "").strip()
        numtaps = int(numtaps_raw) if numtaps_raw else None

        # parse edges depending on filter type
        if ftype == "lowpass":
            edges = [float(request.form.get("f_pass")), float(request.form.get("f_stop"))]
        elif ftype == "highpass":
            print("FORM DATA:", request.form)
            edges = [float(request.form.get("f_stop")), float(request.form.get("f_pass"))]
            print(edges)
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

        sine_freqs = [
            float(x.strip())
            for x in request.form.get("sine_freqs", "1000").split(",")
            if x.strip()
        ]

        # Where plots are saved (dsp_full_flow writes to plots/) - ensure static path exists
        os.makedirs("plots", exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)

        # 2. Run Sine generator (non-interactive)
        t, wave = run_sine(fs, sine_freqs, duration=1.0)
        # copy plot to static/plots
        if os.path.exists("plots/sine_input_plot_zoomed.png"):
            subprocess.run(f"cp plots/sine_input_plot_zoomed.png {PLOTS_DIR}/sine_input_plot_zoomed.png", shell=True)

        # 3. FIR design & save coeffs (Q5.27)
        taps = design_fir(fs, ripple, attenuation, ftype, edges, numtaps)
        save_coeffs_q5_27(taps, path="output/coeff.txt")
        # plot filter response same style as fir_design.py
        plot_filter_response(fs, taps, ftype, edges, ripple, attenuation, numtaps)
        if os.path.exists("plots/filter_response.png"):
            subprocess.run(f"cp plots/filter_response.png {PLOTS_DIR}/filter_response.png", shell=True)

        # 4. Run simulation using your bash function (runs interactive bash to source function)
        # this will block until done
        run_verilog_sim_with_bash_function()

        # 5. Analyze Verilog output: writes into plots/
        verilog_y = analyze_verilog_output(fs, in_path="input/filter_response_verilog.txt")
        # copy Verilog plots to static
        for src in ("filter_out_fft_verilog.png", "filter_out_time_verilog.png"):
            if os.path.exists(f"plots/{src}"):
                subprocess.run(f"cp plots/{src} {PLOTS_DIR}/{src}", shell=True)

        # 6. Ideal response (python) and plots
        ideal_y = run_ideal_response(fs, taps, sine_freqs, duration=1.0)
        for src in ("ideal_response_time.png", "ideal_response_fft.png"):
            if os.path.exists(f"plots/{src}"):
                subprocess.run(f"cp plots/{src} {PLOTS_DIR}/{src}", shell=True)

        # 7. Compare (RMSE + comparison plots)
        rmse = compare_ideal_vs_verilog(fs)
        for src in ("comparison_time.png", "comparison_fft.png"):
            if os.path.exists(f"plots/{src}"):
                subprocess.run(f"cp plots/{src} {PLOTS_DIR}/{src}", shell=True)

        # Pass results to show page
        results = {
            "fs": fs,
            "ftype": ftype,
            "ripple": ripple,
            "attenuation": attenuation,
            "edges": edges,
            "sine_freqs": sine_freqs,
            "numtaps": len(taps),
            "rmse": float(rmse),
        }
        return render_template("results.html", results=results, plots_dir=f"/static/plots")
    except Exception as e:
        trace = traceback.format_exc()
        flash(f"Error during full flow: {e}\nSee server logs for details.")
        print(trace)
        return redirect(url_for("index"))

if __name__ == "__main__":
    # For development only: set debug=False when running on shared machine
    app.run(host="0.0.0.0", port=5000, debug=False)

