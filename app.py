
import os
import io
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from agent.graph import build_graph
from agent.state import RLDAState

st.set_page_config(page_title="RLDA Agent (Quarter-Car)", layout="wide")
st.title("🚗 RLDA Agent – Quarter-Car (LangGraph + GPT‑4o‑mini)")

st.markdown(
    """
This app runs the full **agentic workflow** (Planner → Validator → Road Prep → Simulation → Analysis → Report → Persist)
for **Road Load Data (RLDA)** generation using a quarter-car vertical model.

**Tips**
- Provide your **suspension parameters** (CSV). Optionally provide **damper dyno** and a **measured road**.
- If no road is provided, the agent synthesizes an **ISO roughness** road.
- Heavy physics is executed by the simulator tool – the LLM only plans & orchestrates.
    """
)

# --- Sidebar Inputs ---
st.sidebar.header("Inputs")
# API key check (uses env var)
openai_key = os.getenv("OPENAI_API_KEY", "")
if not openai_key:
    st.sidebar.warning("OPENAI_API_KEY not set. Set the env var for GPT-4o-mini access.")

# Uploaders
params_file = st.sidebar.file_uploader("Suspension Params CSV", type=["csv"], help="Required: columns corner_id,m_s,m_u,k_spring,c_bump,c_rebound,k_tire ...")
damper_file = st.sidebar.file_uploader("Damper Dyno CSV (optional)", type=["csv"], help="Columns: v_mps,F_N")
road_file = st.sidebar.file_uploader("Road Profile CSV (optional)", type=["csv"], help="Columns: t,z_r")

# Numeric controls
fs = st.sidebar.number_input("Sample Rate (Hz)", min_value=100, max_value=5000, value=1000, step=100)
duration = st.sidebar.number_input("Duration (s)", min_value=5, max_value=120, value=20, step=5)
iso_class = st.sidebar.selectbox("ISO Class (if no measured road)", ["A","B","C","D","E","G","H"], index=2)
seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=10_000, value=42, step=1)

use_damper_map = st.sidebar.checkbox("Use damper dyno map", value=False)

m = st.sidebar.number_input("Wöhler exponent m", min_value=1.0, max_value=12.0, value=6.0, step=0.5)
Nref = st.sidebar.number_input("Reference cycles N_ref", min_value=1e3, max_value=1e8, value=1e6, step=1e5, format="%.0f")

# Out dir per run
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = f"out/streamlit_{run_id}"

# Helper: materialize uploaded file to disk
WORK = Path("work"); WORK.mkdir(parents=True, exist_ok=True)

def save_upload(uploaded, default_path: Path) -> str:
    if uploaded is None:
        return ""
    default_path.parent.mkdir(parents=True, exist_ok=True)
    with open(default_path, 'wb') as f:
        f.write(uploaded.read())
    return str(default_path)

# Discover corners when params is provided
corner_id = None
if params_file is not None:
    df_tmp = pd.read_csv(params_file)
    if 'corner_id' in df_tmp.columns:
        corner_id = st.sidebar.selectbox("Corner", list(df_tmp['corner_id'].unique()), index=0)

# Run button
run = st.button("▶️ Run Agent")

if run:
    # Save uploads
    params_path = save_upload(params_file, WORK / f"suspension_{run_id}.csv")
    damper_path = save_upload(damper_file, WORK / f"damper_{run_id}.csv") if use_damper_map else ""
    road_path = save_upload(road_file, WORK / f"road_{run_id}.csv")

    if not params_path:
        st.error("Please upload a Suspension Params CSV.")
        st.stop()

    # Build state
    state = RLDAState(
        params_csv=params_path,
        corner_id=corner_id,
        road_csv=road_path if road_path else None,
        fs_hz=float(fs),
        duration_s=float(duration),
        iso_class=iso_class,
        seed=int(seed),
        use_damper_map=bool(use_damper_map and bool(damper_path)),
        damper_map_csv=damper_path if (use_damper_map and damper_path) else None,
        wohler_m=float(m),
        N_ref=float(Nref),
        out_dir=out_dir
    )

    with st.spinner("Building agent and executing workflow..."):
        app = build_graph().compile()
        try:
            final = app.invoke(state)
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success("Agent run complete")

    # Load artifacts
    artifacts = final.artifacts or {}
    metrics_path = artifacts.get('metrics')
    time_hist_path = artifacts.get('time_history')
    psd_path = artifacts.get('psd')
    rf_path = artifacts.get('rainflow')
    report_md_path = artifacts.get('report_md')

    cols = st.columns(3)
    with cols[0]:
        st.subheader("KPIs")
        if metrics_path and os.path.exists(metrics_path):
            metrics = pd.read_json(metrics_path, typ='series')
            st.json(metrics.to_dict())
        else:
            st.info("No metrics found.")

    with cols[1]:
        st.subheader("Report")
        if report_md_path and os.path.exists(report_md_path):
            st.markdown(Path(report_md_path).read_text())
        else:
            st.info("No report generated.")

    with cols[2]:
        st.subheader("Artifacts")
        st.write(artifacts)

    st.divider()

    # Plots
    st.subheader("Time History: Fz")
    if time_hist_path and os.path.exists(time_hist_path):
        df = pd.read_csv(time_hist_path)
        fig = px.line(df, x='t', y='Fz', title='Wheel Vertical Load (Fz)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Time history not available.")

    st.subheader("PSD of Fz")
    if psd_path and os.path.exists(psd_path):
        psd = pd.read_csv(psd_path)
        fig = px.line(psd, x='f_Hz', y='PSD_N2_per_Hz', title='PSD of Fz')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("PSD not available.")

    st.subheader("Rainflow Ranges (Histogram)")
    if rf_path and os.path.exists(rf_path):
        rf = pd.read_csv(rf_path)
        fig = px.histogram(rf, x='range_N', nbins=50, title='Rainflow Ranges (N)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Rainflow data not available.")

    # Download bundle
    st.divider()
    st.subheader("Download Results")

    def zipdir(path):
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
            p = Path(path)
            for fp in p.rglob('*'):
                z.write(fp, fp.relative_to(p))
        mem.seek(0)
        return mem

    if os.path.isdir(out_dir):
        zbuf = zipdir(out_dir)
        st.download_button("Download run folder (ZIP)", data=zbuf, file_name=f"rlda_run_{run_id}.zip")
