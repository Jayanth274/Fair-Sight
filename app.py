import streamlit as st
import pandas as pd
import os
import time
from modules.detector import scan_columns, compute_bias_metrics
from modules.explainer import explain_bias
from modules.fixer import apply_reweighing, apply_postprocessing
from modules.reporter import generate_pdf_report, generate_fairness_certificate
import plotly.express as px
import requests

# 1. Page Configuration
st.set_page_config(
    page_title="FairSight",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Load custom CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# 3. Session State Initialization
if "df" not in st.session_state:
    st.session_state.df = None
if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

def generate_gemini_insight(sensitive_col, target_col, fairness_definition, di, spd, eod, verdict):
    """Calls Gemini REST API to get a plain English explanation of fairness results."""
    try:
        api_key = st.secrets['GEMINI_API_KEY']
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        prompt = f"You are a fairness expert. A bias analysis was run. Sensitive attribute: {sensitive_col}. Target column: {target_col}. Fairness definition: {fairness_definition}. Disparate Impact={di:.3f}, Statistical Parity Difference={spd:.3f}, Equal Opportunity Difference={eod:.3f}. Overall verdict: {verdict}. In 3 sentences maximum, explain what this means for a business decision maker. If biased mention real world consequences. If fair reassure the user."
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(url, json=payload)
        result = response.json()
        print("GEMINI RESPONSE:", result)
        if 'candidates' in result:
            return result['candidates'][0]['content']['parts'][0]['text']
        elif 'error' in result:
            return f"Gemini error: {result['error']['message']}"
        else:
            return f"Unexpected response: {str(result)}"
    except Exception as e:
        return f"AI Insight unavailable: {str(e)}"

def main():
    # Header Section
    st.markdown("<h1 class='fairsight-logo'>FairSight</h1>", unsafe_allow_html=True)
    st.markdown("<p class='fairsight-tagline'>The world's first bias detection platform that understands YOUR definition of fairness — not ours.</p>", unsafe_allow_html=True)
    
    # State routing
    current_phase = st.session_state.get("current_phase", 1 if st.session_state.df is None else 3)
    
    if current_phase == 1:
        render_landing_page()
    else:
        # Global Back Button
        col_b1, col_b2 = st.columns([1, 10])
        with col_b1:
            target_phase = 1 if current_phase == 3 else current_phase - 1
            if st.button("⬅️ Back"):
                st.session_state.current_phase = target_phase
                st.rerun()
                
        if current_phase == 3:
            render_phase3()
        elif current_phase == 4:
            render_phase4()
        elif current_phase == 5:
            render_phase5()
        elif current_phase == 6:
            render_phase6()
        elif current_phase == 7:
            render_phase7()

def render_landing_page():
    # Particles Background
    st.markdown("""
    <ul class="particles">
        <li class="particle"></li><li class="particle"></li><li class="particle"></li>
        <li class="particle"></li><li class="particle"></li><li class="particle"></li>
        <li class="particle"></li><li class="particle"></li><li class="particle"></li>
        <li class="particle"></li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: #9CA3AF; margin-bottom: 2rem; font-weight: 400;'>How it works</h3>", unsafe_allow_html=True)
    # Three animated info cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-card fade-in" style="animation-delay: 0.1s;">
            <div class="info-card-icon">🔍</div>
            <div class="info-card-title">Detect</div>
            <div class="info-card-text">Automatically scan your dataset to uncover potential sensitive attributes and predictive targets without writing a single line of code.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="info-card fade-in" style="animation-delay: 0.3s;">
            <div class="info-card-icon">📏</div>
            <div class="info-card-title">Measure</div>
            <div class="info-card-text">Evaluate your model against multiple industry-standard fairness definitions, tailored to the unique context of your data and use case.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="info-card fade-in" style="animation-delay: 0.5s;">
            <div class="info-card-icon">🛠️</div>
            <div class="info-card-title">Fix</div>
            <div class="info-card-text">Apply state-of-the-art debiasing algorithms at the pre-processing, in-processing, or post-processing stages to mitigate bias.</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # File Upload UI
    st.markdown("<h3 style='text-align: center; margin-bottom: 1rem;'>👇 Upload your dataset below to begin</h3>", unsafe_allow_html=True)
    
    # Centering the file uploader using columns
    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        uploaded_file = st.file_uploader("", type=["csv"], help="Upload a tabular dataset in CSV format.")
        
        if uploaded_file is not None:
            # Check if this is a NEW file (different from previous)
            if uploaded_file.name != st.session_state.get('uploaded_filename', ''):
                # Clear all session state EXCEPT current_phase
                for key in list(st.session_state.keys()):
                    if key != 'current_phase':
                        del st.session_state[key]
                st.session_state.uploaded_filename = uploaded_file.name
                st.rerun()
            
            # Process the file if not already loaded in state
            if "df" not in st.session_state or st.session_state.df is None:
                with st.spinner("Scanning your dataset..."):
                    try:
                        time.sleep(1) # Tiny sleep for UI feedback
                        df = pd.read_csv(uploaded_file)
                        st.session_state.df = df
                        
                        # Call scan_columns and move to Phase 3
                        scan_results = scan_columns(df)
                        st.session_state.scan_results = scan_results
                        st.session_state.current_phase = 3
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📖 How to use FairSight")
    
    with st.expander("Step 1 — Upload Your Dataset", expanded=False):
        st.write("""
        Accept any CSV file. FairSight automatically scans for sensitive attributes like **gender, race, age** and detects your **target prediction column**. 
        Our intelligent scanner identifies candidate columns based on keyword matching and data distribution patterns, ensuring no manual coding is required to start your analysis.
        """)
        
    with st.expander("Step 2 — Define What Fairness Means to You", expanded=False):
        st.markdown("""
        Choose a fairness definition per sensitive attribute. Every definition represents a different philosophy of justice:
        
        *   **Demographic Parity**: Ensures equal outcome rates across different groups, regardless of qualifications. 
            *   *Measured by:* Statistical Parity Difference (SPD).
            *   *Formula:* `P(Y=1|A=0) - P(Y=1|A=1)`
        *   **Equal Opportunity**: Ensures that qualified individuals from all groups have an equal chance of receiving a favorable outcome (equal true positive rates).
            *   *Measured by:* Equal Opportunity Difference (EOD).
            *   *Formula:* `TPR(A=0) - TPR(A=1)`
        *   **Equalized Odds**: A strict definition ensuring both equal true positive rates AND equal false positive rates across groups.
            *   *Measured by:* Both SPD and EOD simultaneously.
        """)
        
    with st.expander("Step 3 — Run Bias Analysis", expanded=False):
        st.markdown("""
        FairSight computes three core metrics to quantify bias:
        *   **Disparate Impact (DI)**: The ratio of favorable outcomes between groups. *Fair range: 0.8 to 1.2.*
        *   **Statistical Parity Difference (SPD)**: The absolute difference in outcome rates. *Fair range: -0.1 to 0.1.*
        *   **Equal Opportunity Difference (EOD)**: The difference in true positive rates. *Fair range: -0.1 to 0.1.*
        
        **SHAP explainability** identifies which features drive bias and detects **proxy variables** — hidden features (like Zip Code) that may be acting as substitutes for protected sensitive attributes.
        """)
        
    with st.expander("Step 4 — Fix and Download", expanded=False):
        st.markdown("""
        Choose a mitigation strategy based on where in the pipeline you want to intervene:
        *   **Reweighing (Pre-processing)**: Assigns higher weights to underrepresented groups before training. Download the reweighed dataset and retrain your model using the `instance_weight` column.
        *   **Reject Option Classification (Post-processing)**: Flips borderline predictions to favor the unprivileged group after the model has made its prediction.
        
        Once fixed, you can download your **debiased dataset**, **fair model (.pkl)**, and a **professional PDF report**.
        """)

def render_phase3():
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Dataset Intelligence Report</h2>", unsafe_allow_html=True)
    
    scan_results = st.session_state.scan_results
    overview = scan_results["dataset_overview"]
    df = st.session_state.df
    
    # Section A: Dataset Overview
    st.markdown("### 📊 Dataset Overview")
    
    # Calculate Dataset Health Score
    num_sensitive = len(scan_results['sensitive_candidates'])
    total_missing = sum(overview['missing_values'].values())
    
    # New Formula: -8 per sensitive col, -2 per 500 missing values
    missing_penalty = (total_missing // 500) * 2
    health_score = max(0, min(100, 100 - (8 * num_sensitive) - missing_penalty))
    
    if health_score > 70:
        health_color = "#10B981" # Green
    elif health_score >= 40:
        health_color = "#F59E0B" # Amber
    else:
        health_color = "#EF4444" # Red
        
    st.markdown(f"<h3 style='text-align: center; margin-bottom: 2rem;'>Health Score: <span style='color: {health_color}; font-size: 3rem; font-weight: 800;'>{health_score:.0f}</span> <span style='color: #9CA3AF; font-size: 1.5rem;'>/ 100</span></h3>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ How is this calculated?"):
        st.markdown("The score starts at **100**. We subtract **8 points** for each sensitive column detected, and **2 points** for every **500 missing values** in the dataset.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{overview['row_count']:,}")
    with col2:
        st.metric("Total Columns", f"{overview['column_count']:,}")
    with col3:
        total_missing = sum(overview['missing_values'].values())
        st.metric("Missing Values", f"{total_missing:,}")
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sensitivity saturation warning
    if overview['column_count'] > 0 and (num_sensitive / overview['column_count']) > 0.5:
        st.warning("⚠️ **Warning:** Many columns were auto-detected as potentially sensitive based on cardinality. This dataset may use generic column names. Please manually review and deselect columns that are not actually sensitive attributes using the Perspective Engine below.")
    
    st.markdown("#### Column Profiling")
    prof_data = []
    for col, dtype in overview["data_types"].items():
        types = []
        if col in scan_results['sensitive_candidates']:
            types.append("SENSITIVE")
        if col in scan_results['target_candidates']:
            types.append("TARGET")
            
        if pd.api.types.is_numeric_dtype(df[col]):
            types.append("NUMERIC")
        else:
            types.append("CATEGORICAL")
            
        missing = overview["missing_values"].get(col, 0)
        prof_data.append({
            "Column Name": col,
            "Data Type": str(dtype),
            "Missing Values": missing,
            "Flags": ", ".join(types)
        })
        
    st.dataframe(pd.DataFrame(prof_data), use_container_width=True, hide_index=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Section B: Perspective Engine
    st.markdown("### 🎯 Perspective Engine")
    st.markdown("<p style='color: #9CA3AF;'>Select which columns might contain bias and define what fairness means for each.</p>", unsafe_allow_html=True)
    
    # Target Selection
    st.markdown("#### 1. Define the Target (Prediction Outcome)")
    all_columns = list(df.columns)
    default_target = scan_results["target_candidates"][0] if scan_results["target_candidates"] else all_columns[0]
    
    selected_target = st.selectbox(
        "Select the column your model will predict:",
        options=all_columns,
        index=all_columns.index(default_target) if default_target in all_columns else 0
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 2. Identify Sensitive Attributes")
    
    if "current_candidates" not in st.session_state:
        st.session_state.current_candidates = list(scan_results["sensitive_candidates"])
        
    if not st.session_state.current_candidates:
        st.info("ℹ️ **No sensitive attributes were automatically detected in this dataset.** This may indicate your dataset does not contain demographic or protected attributes, or column names are not recognizable. You can manually add columns for analysis using the selector below.")
    
    # Manual Add
    available_to_add = [c for c in all_columns if c not in st.session_state.current_candidates and c != selected_target and pd.notna(c) and str(c).strip().lower() not in ["none", "null", "nan", ""]]
    
    col_add1, col_add2 = st.columns([2, 1])
    with col_add1:
        col_to_add = st.selectbox("Missed a column? Add it manually:", ["(Select...)"] + available_to_add)
        if col_to_add != "(Select...)":
            st.session_state.current_candidates.append(col_to_add)
            st.session_state[f"chk_{col_to_add}"] = True 
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    confirmed_cols = []
    
    for col in st.session_state.current_candidates:
        c1, c2 = st.columns([1, 2])
        with c1:
            if f"chk_{col}" not in st.session_state:
                st.session_state[f"chk_{col}"] = (col in scan_results["sensitive_candidates"])
                
            is_checked = st.checkbox(col, key=f"chk_{col}")
            if is_checked:
                confirmed_cols.append(col)
                
        with c2:
            if is_checked:
                if f"def_{col}" not in st.session_state:
                    st.session_state[f"def_{col}"] = "Demographic Parity"
                st.selectbox(
                    f"Fairness definition for {col}:",
                    ["Demographic Parity", "Equal Opportunity", "Equalized Odds"],
                    key=f"def_{col}"
                )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    can_run = len(confirmed_cols) > 0 and selected_target is not None
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Run Bias Analysis", disabled=not can_run, type="primary"):
            st.session_state.target_col = selected_target
            st.session_state.sensitive_cols = confirmed_cols
            st.session_state.fairness_definitions = {col: st.session_state[f"def_{col}"] for col in confirmed_cols}
            st.session_state.current_phase = 4
            st.rerun()
            
    with col_btn2:
        if st.button("Reset / Upload New File"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def render_phase4():
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Bias Dashboard</h2>", unsafe_allow_html=True)
    
    df = st.session_state.df
    target_col = st.session_state.target_col
    
    # Run computations if not done yet
    if "bias_metrics" not in st.session_state:
        metrics_dict = {}
        for s_col in st.session_state.sensitive_cols:
            fair_def = st.session_state.fairness_definitions[s_col]
            with st.spinner(f"Computing metrics for {s_col}..."):
                res = compute_bias_metrics(df, s_col, target_col, fair_def)
                if "error" in res:
                    st.error(f"🚨 **Analysis Error:** {res['error']}")
                    if st.button("Go Back to Dataset Intelligence"):
                        st.session_state.current_phase = 3
                        st.rerun()
                    st.stop()
                metrics_dict[s_col] = res
        st.session_state.bias_metrics = metrics_dict

    # Certificate logic is now per-tab below

    tabs = st.tabs(st.session_state.sensitive_cols)
    
    def render_metric_card(title, value, verdict, fair_range, explanation, color, pct_dist):
        return f"""
        <div class="info-card fade-in">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin:0; font-size:1.1rem; color: #9CA3AF;">{title} <span title="{explanation}">ℹ️</span></h4>
                <span style="color: {color}; font-weight: bold; font-size: 0.8rem;">{verdict}</span>
            </div>
            <div style="font-size: 2.5rem; font-weight: 800; color: #F9FAFB; margin: 0.5rem 0;">{value:.3f}</div>
            <div style="font-size: 0.85rem; color: #6B7280; margin-bottom: 0.5rem;">Fair Range: {fair_range}</div>
            <div class="prog-bar-bg">
                <div style="width: {min(100, max(0, pct_dist))}%; height: 100%; background-color: {color}; border-radius: 5px; transition: width 0.5s ease-in-out;"></div>
            </div>
        </div>
        """
        
    for idx, s_col in enumerate(st.session_state.sensitive_cols):
        with tabs[idx]:
            metrics = st.session_state.bias_metrics[s_col]
            
            # Verdict bar
            overall_verdict = metrics["overall_verdict"]
            bg_color = "#10B981" if overall_verdict == "FAIR" else "#EF4444"
            pulse_class = "pulse-danger" if overall_verdict == "BIASED" else ""
            overall_text = "BIASED based on your fairness definition" if overall_verdict == "BIASED" else "FAIR based on your fairness definition"
            
            st.markdown(f"""
            <div class="{pulse_class}" style="background-color: {bg_color}; padding: 1rem; border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                <h3 style="color: white; margin: 0; font-weight: 800;">{overall_text}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if overall_verdict == "FAIR":
                st.markdown("<h3 style='text-align: center; color: #10B981; margin-top: 1rem;'>✨ Congratulations! ✨</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; color: #9CA3AF;'>Your dataset meets your defined fairness criteria for <b>{s_col}</b>. No bias mitigation required for this attribute.</p>", unsafe_allow_html=True)
                
                # Certificate Card
                metadata = {
                    'filename': st.session_state.uploaded_filename if "uploaded_filename" in st.session_state else "dataset.csv",
                    'definitions': st.session_state.fairness_definitions
                }
                
                # Show DI, SPD, EOD in certificate
                di_val, spd_val, eod_val = metrics["DI"]["value"], metrics["SPD"]["value"], metrics["EOD"]["value"]
                di_stat, spd_stat, eod_stat = metrics["DI"]["verdict"], metrics["SPD"]["verdict"], metrics["EOD"]["verdict"]
                
                cert_html = f"""
                <div class="info-card fade-in" style="border: 2px solid #10B981; background: #0E1A14; max-width: 700px; margin: 1.5rem auto; padding: 2rem; border-radius: 20px; box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);">
                    <h2 style="text-align: center; color: #10B981; margin-top: 0; font-size: 1.5rem;">FairSight Fairness Certificate</h2>
                    <hr style="border-color: #10B981; opacity: 0.2;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.5rem 0; font-size: 0.9rem;">
                        <div><p style="color: #9CA3AF; margin:0;">Dataset</p><p style="font-weight:bold;">{metadata['filename']}</p></div>
                        <div><p style="color: #9CA3AF; margin:0;">Attribute</p><p style="font-weight:bold;">{s_col}</p></div>
                        <div><p style="color: #9CA3AF; margin:0;">Definition</p><p style="font-weight:bold;">{st.session_state.fairness_definitions[s_col]}</p></div>
                        <div><p style="color: #9CA3AF; margin:0;">Date</p><p style="font-weight:bold;">{time.strftime("%B %d, %Y")}</p></div>
                    </div>
                    <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem;">
                        <p style="margin:0 0 0.5rem 0; font-weight:bold; font-size:0.8rem; color:#9CA3AF;">METRICS SUMMARY</p>
                        <div style="display:flex; justify-content:space-between; font-family:monospace;">
                            <span>DI: {di_val:.3f} ({di_stat})</span>
                            <span>SPD: {spd_val:.3f} ({spd_stat})</span>
                            <span>EOD: {eod_val:.3f} ({eod_stat})</span>
                        </div>
                    </div>
                    <p style="color: #10B981; font-weight: bold; font-size: 1.4rem; text-align:center; letter-spacing: 3px; margin:0;">CERTIFIED FAIR</p>
                </div>
                """
                st.markdown(cert_html, unsafe_allow_html=True)
                
                col_c1, col_c2, col_c3 = st.columns([1, 2, 1])
                with col_c2:
                    # Pass only the current column metrics to the generator for a targeted certificate
                    with st.spinner("Generating Certificate..."):
                        cert_bytes = generate_fairness_certificate(metadata, {s_col: metrics})
                        
                    st.download_button(
                        label=f"🏆 Download Fairness Certificate ({s_col})",
                        data=cert_bytes,
                        file_name=f"FairSight_Certificate_{s_col}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                        key=f"cert_dl_{s_col}"
                    )
                    
                    if st.button("🔄 Test Another Dataset", use_container_width=True, key=f"reset_{s_col}"):
                        for key in list(st.session_state.keys()):
                            del st.session_state[key]
                        st.rerun()
                
                st.markdown("<p style='text-align: center; color: #9CA3AF; margin-top: 2rem;'>Your dataset meets fairness criteria. You may still explore mitigation strategies below.</p>", unsafe_allow_html=True)
                st.markdown("<br><hr><br>", unsafe_allow_html=True)

            
            # Dynamic verdict sentence
            fair_def = st.session_state.fairness_definitions[s_col]
            di = metrics["DI"]
            spd = metrics["SPD"]
            eod = metrics["EOD"]
            
            if overall_verdict == "BIASED":
                if fair_def == "Demographic Parity":
                    st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #FCA5A5;'>Your model shows a Statistical Parity Difference of {spd['value']:.3f} which exceeds the fair threshold of 0.1 — meaning the privileged group receives favorable outcomes significantly more often.</p>", unsafe_allow_html=True)
                elif fair_def == "Equal Opportunity":
                    st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #FCA5A5;'>Your model shows an Equal Opportunity Difference of {eod['value']:.3f} which exceeds the fair threshold of 0.1 — meaning true positive rates are significantly unbalanced.</p>", unsafe_allow_html=True)
                else:
                    spd_biased = spd["verdict"] == "BIASED"
                    eod_biased = eod["verdict"] == "BIASED"
                    
                    if spd_biased and eod_biased:
                        st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #FCA5A5;'>Your model violates Equalized Odds — both Statistical Parity (SPD: {spd['value']:.3f}) and Equal Opportunity (EOD: {eod['value']:.3f}) exceed their fair thresholds.</p>", unsafe_allow_html=True)
                    elif spd_biased:
                        st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #FCA5A5;'>Your model violates Equalized Odds because the Statistical Parity Difference ({spd['value']:.3f}) exceeds the fair threshold of 0.1.</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #FCA5A5;'>Your model violates Equalized Odds because the Equal Opportunity Difference ({eod['value']:.3f}) exceeds the fair threshold of 0.1.</p>", unsafe_allow_html=True)
            else:
                if fair_def == "Demographic Parity":
                    st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #6EE7B7;'>Your model meets the Demographic Parity threshold — favorable outcomes are distributed evenly across groups.</p>", unsafe_allow_html=True)
                elif fair_def == "Equal Opportunity":
                    st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #6EE7B7;'>Your model meets the Equal Opportunity threshold — true positive rates are balanced across groups.</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #6EE7B7;'>Your model meets the Equalized Odds threshold — both outcome rates and error rates are balanced.</p>", unsafe_allow_html=True)
            
            # AI Insight Section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🤖 AI Insight")
            
            insight_key = f"insight_{s_col}"
            if insight_key not in st.session_state:
                with st.spinner("AI is analyzing your results..."):
                    insight = generate_gemini_insight(
                        s_col, target_col, fair_def, 
                        di['value'], spd['value'], eod['value'], overall_verdict
                    )
                    st.session_state[insight_key] = insight
            
            st.markdown(f"""
            <div style="background-color: rgba(124, 58, 237, 0.1); border: 1px solid #7C3AED; padding: 1.5rem; border-radius: 10px; color: #F9FAFB;">
                <p style="margin: 0; font-style: italic;">"{st.session_state[insight_key]}"</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            di_color = "#10B981" if di["verdict"] == "FAIR" else "#EF4444"
            spd_color = "#10B981" if spd["verdict"] == "FAIR" else "#EF4444"
            eod_color = "#10B981" if eod["verdict"] == "FAIR" else "#EF4444"
            
            # Animate metric cards using st.empty
            metrics_placeholder = st.empty()
            
            # Only animate once per column session
            anim_key = f"anim_done_{s_col}"
            if anim_key not in st.session_state:
                frames = 15
                for i in range(frames + 1):
                    progress = i / frames
                    c_di = di["value"] * progress
                    c_spd = spd["value"] * progress
                    c_eod = eod["value"] * progress
                    
                    c_di_pct = (min(abs(c_di), 2.0) / 2.0) * 100
                    c_spd_pct = 50 + (max(-1, min(1, c_spd)) * 50)
                    c_eod_pct = 50 + (max(-1, min(1, c_eod)) * 50)
                    
                    html_str = f"""
                    <div style="display:flex; gap: 1rem;">
                        <div style="flex: 1;">{render_metric_card("Disparate Impact", c_di, di["verdict"], "0.8 - 1.2", "Ratio of favorable outcomes.", di_color, c_di_pct)}</div>
                        <div style="flex: 1;">{render_metric_card("Statistical Parity", c_spd, spd["verdict"], "-0.1 - 0.1", "Difference in rate of favorable outcomes.", spd_color, c_spd_pct)}</div>
                        <div style="flex: 1;">{render_metric_card("Equal Opportunity", c_eod, eod["verdict"], "-0.1 - 0.1", "Difference in true positive rates.", eod_color, c_eod_pct)}</div>
                    </div>
                    """
                    metrics_placeholder.markdown(html_str, unsafe_allow_html=True)
                    time.sleep(0.03)
                st.session_state[anim_key] = True
            else:
                c_di_pct = (min(abs(di["value"]), 2.0) / 2.0) * 100
                c_spd_pct = 50 + (max(-1, min(1, spd["value"])) * 50)
                c_eod_pct = 50 + (max(-1, min(1, eod["value"])) * 50)
                
                html_str = f"""
                <div style="display:flex; gap: 1rem;">
                    <div style="flex: 1;">{render_metric_card("Disparate Impact", di["value"], di["verdict"], "0.8 - 1.2", "Ratio of favorable outcomes.", di_color, c_di_pct)}</div>
                    <div style="flex: 1;">{render_metric_card("Statistical Parity", spd["value"], spd["verdict"], "-0.1 - 0.1", "Difference in rate of favorable outcomes.", spd_color, c_spd_pct)}</div>
                    <div style="flex: 1;">{render_metric_card("Equal Opportunity", eod["value"], eod["verdict"], "-0.1 - 0.1", "Difference in true positive rates.", eod_color, c_eod_pct)}</div>
                </div>
                """
                metrics_placeholder.markdown(html_str, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Plotly Chart
            st.markdown(f"### Outcome Rates by `{s_col}`")
            plot_df = df.copy()
            plot_df[target_col] = plot_df[target_col].astype(str).str.strip().str.rstrip('.')
            grouped = plot_df.groupby([s_col, target_col]).size().reset_index(name="Count")
            
            fig = px.bar(grouped, x=s_col, y="Count", color=target_col, barmode="group",
                         color_discrete_sequence=["#7C3AED", "#10B981", "#F59E0B", "#EF4444"],
                         template="plotly_dark")
            fig.update_layout(
                plot_bgcolor="#1E2130",
                paper_bgcolor="#1E2130"
            )
            st.plotly_chart(fig, use_container_width=True)

    any_fair = any(m["overall_verdict"] == "FAIR" for m in st.session_state.bias_metrics.values())
    if any_fair:
        st.markdown("<p style='text-align: center; color: #9CA3AF;'>Your dataset meets fairness criteria. You may still explore mitigation strategies below.</p>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Proceed to SHAP Explainability", type="primary"):
            st.session_state.current_phase = 5
            st.rerun()
            
    with col_btn2:
        if st.button("Reset / Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def render_phase5():
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>SHAP Explainability & Proxy Detection</h2>", unsafe_allow_html=True)
    
    df = st.session_state.df
    target_col = st.session_state.target_col
    
    if "shap_results" not in st.session_state:
        # We run the explainer. The proxy detection checks against the first sensitive column.
        s_col = st.session_state.sensitive_cols[0]
        with st.spinner("Training model & computing SHAP values (this may take a moment)..."):
            res = explain_bias(df, s_col, target_col)
            st.session_state.shap_results = res
            st.session_state.shap_s_col = s_col

    res = st.session_state.shap_results
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Feature Impact (SHAP Summary)")
        st.image(res["shap_plot_path"], use_container_width=True)
        
        # Auto-generated plain English interpretation
        top_feature = res["top_features"][0]["feature"]
        st.markdown(f"<p style='color: #9CA3AF; margin-top: 1rem;'><strong>💡 Interpretation:</strong> <code>{top_feature}</code> is the strongest driver of predictions in your model. If this feature is acting as a proxy for <code>{st.session_state.shap_s_col}</code>, removing the sensitive column alone will not fix the bias.</p>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("### Top Drivers of Predictions")
        for i, feature in enumerate(res["top_features"]):
            st.markdown(f"**{i+1}. {feature['feature']}** (Impact: {feature['importance']:.3f})")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        num_proxies = len(res["proxy_warnings"])
        if num_proxies == 0:
            risk_class = "risk-low"
            risk_text = "LOW"
        elif num_proxies == 1:
            risk_class = "risk-medium"
            risk_text = "MEDIUM"
        else:
            risk_class = "risk-high"
            risk_text = "HIGH"
            
        st.markdown(f"### Proxy Risk: <span class='{risk_class}'>{risk_text}</span>", unsafe_allow_html=True)
        
        if res["proxy_warnings"]:
            for warning in res["proxy_warnings"]:
                st.error(warning)
        else:
            st.success(f"No strong proxy variables detected for '{st.session_state.shap_s_col}'.")
            
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Proceed to Fix Engine", type="primary"):
            st.session_state.current_phase = 6
            st.rerun()
            
    with col_btn2:
        if st.button("Reset / Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def render_phase6():
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>The Fix Engine</h2>", unsafe_allow_html=True)
    
    df = st.session_state.df
    target_col = st.session_state.target_col
    s_col = st.session_state.sensitive_cols[0]
    fairness_def = st.session_state.fairness_definitions[s_col]
    
    st.markdown(f"<p style='text-align: center; color: #9CA3AF;'>Configured to mitigate bias for <b>{s_col}</b> using <b>{fairness_def}</b>.</p>", unsafe_allow_html=True)
    
    # Current Metrics Badges
    curr_metrics = st.session_state.bias_metrics[s_col]
    di_val = curr_metrics["DI"]["value"]
    spd_val = curr_metrics["SPD"]["value"]
    eod_val = curr_metrics["EOD"]["value"]
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <span class='badge badge-categorical' style='font-size: 1rem;'>Current DI: {di_val:.3f}</span>
        <span class='badge badge-categorical' style='font-size: 1rem;'>Current SPD: {spd_val:.3f}</span>
        <span class='badge badge-categorical' style='font-size: 1rem;'>Current EOD: {eod_val:.3f}</span>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Pre-processing", "In-processing", "Post-processing"])
    
    with tab1:
        st.markdown("### Pre-processing: Reweighing")
        st.markdown("""
        **How it works:** This method alters the weights of different training examples before the model is even trained. 
        It assigns higher weights to unprivileged groups with favorable outcomes, and lower weights to privileged groups with favorable outcomes. 
        This effectively balances the dataset without actually deleting or creating new data points.
        """)
        
        if st.button("Apply Reweighing", key="btn_reweighing", type="primary"):
            with st.spinner("Applying Reweighing algorithm..."):
                res = apply_reweighing(df, s_col, target_col, fairness_def)
                st.session_state.fixed_metrics = res
                st.success("Reweighing applied successfully!")
                
    with tab2:
        st.markdown("### In-processing: Adversarial Debiasing")
        st.markdown("""
        **How it works:** This method uses a neural network architecture with two competing parts: a predictor trying to guess the target outcome, and an adversary trying to guess the sensitive attribute from the predictor's outputs. The predictor is penalized if the adversary succeeds, forcing it to make predictions that are completely independent of the sensitive attribute.
        """)
        
        st.error("⚠️ **Adversarial Debiasing requires TensorFlow which conflicts with current dependencies. This feature is available in the Phase 2 roadmap.**")
        
    with tab3:
        st.markdown("### Post-processing: Reject Option & Equalized Odds")
        
        pp_method = st.radio("Select Post-Processing Technique:", ["Reject Option Classification", "Calibrated Equalized Odds"])
        
        if pp_method == "Reject Option Classification":
            st.markdown("""
            **How it works:** This method looks at predictions near the decision boundary (where the model is least confident). It flips unfavorable predictions to favorable for the unprivileged group, and vice versa.
            """)
        else:
            st.markdown("""
            **How it works:** This method adjusts the probabilities of the model's output specifically to equalize true positive rates and false positive rates between groups (Equalized Odds).
            """)
            
        if st.button(f"Apply {pp_method}", key="btn_pp", type="primary"):
            with st.spinner(f"Applying {pp_method}..."):
                res = apply_postprocessing(df, s_col, target_col, fairness_def, method=pp_method)
                st.session_state.fixed_metrics = res
                st.success(f"{pp_method} applied successfully!")
                
    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    
    if "fixed_metrics" in st.session_state:
        fixed_res = st.session_state.fixed_metrics
        st.markdown(f"### Results for: {fixed_res['method']}")
        
        if "warning" in fixed_res:
            st.warning(f"💡 {fixed_res['warning']}")
        if "error" in fixed_res:
            st.error(f"🚨 {fixed_res['error']}")
        
        baseline_acc = fixed_res['baseline_accuracy']
        fixed_acc = fixed_res['fixed_accuracy']
        acc_drop = (baseline_acc - fixed_acc) * 100
        
        if acc_drop > 5:
            st.error(f"⚠️ High Accuracy Cost! Baseline: {baseline_acc*100:.2f}% ➡️ Mitigated: {fixed_acc*100:.2f}% (Drop: {acc_drop:.2f}%)")
        elif acc_drop > 2:
            st.warning(f"⚠️ Moderate Accuracy Cost! Baseline: {baseline_acc*100:.2f}% ➡️ Mitigated: {fixed_acc*100:.2f}% (Drop: {acc_drop:.2f}%)")
        else:
            st.success(f"✅ Excellent Accuracy Tradeoff! Baseline: {baseline_acc*100:.2f}% ➡️ Mitigated: {fixed_acc*100:.2f}% (Drop: {max(0, acc_drop):.2f}%)")
        st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        can_proceed = "fixed_metrics" in st.session_state
        if st.button("Proceed to Before vs After Report", disabled=not can_proceed, type="primary"):
            st.session_state.current_phase = 7
            st.rerun()
            
    with col_btn2:
        if st.button("Reset / Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def render_phase7():
    import pickle
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Before vs After Report</h2>", unsafe_allow_html=True)
    
    fixed_res = st.session_state.fixed_metrics
    method = fixed_res["method"]
    before = fixed_res["baseline_metrics"]
    after = fixed_res["fixed_metrics"]
    baseline_acc = fixed_res["baseline_accuracy"]
    fixed_acc = fixed_res["fixed_accuracy"]
    model = fixed_res["model"]
    
    s_col = st.session_state.sensitive_cols[0]
    target_col = st.session_state.target_col
    
    if method == "Reweighing":
        st.warning("⚠️ **Important:** The reweighed dataset contains instance weights for model retraining. Uploading this CSV back into FairSight will show similar bias scores because FairSight measures raw data distributions. The bias improvement shown below reflects predictions from a model trained with these weights — which is the correct way to measure reweighing effectiveness.")
    
    # Calculate % improvements
    di_imp = (abs(1 - before['DI']['value']) - abs(1 - after['DI']['value'])) / abs(1 - before['DI']['value']) * 100 if abs(1 - before['DI']['value']) != 0 else 0
    spd_imp = (abs(before['SPD']['value']) - abs(after['SPD']['value'])) / abs(before['SPD']['value']) * 100 if before['SPD']['value'] != 0 else 0
    eod_imp = (abs(before['EOD']['value']) - abs(after['EOD']['value'])) / abs(before['EOD']['value']) * 100 if before['EOD']['value'] != 0 else 0
    
    avg_imp = max(0, (di_imp + spd_imp + eod_imp) / 3)
    acc_cost = (baseline_acc - fixed_acc) * 100
    
    # Executive Summary
    rec_text = "Recommended for production deployment." if after['overall_verdict'] == 'FAIR' and acc_cost < 5 else "Review accuracy cost before deploying."
    exec_summary = f"{method} reduced overall bias in {s_col} by an average of {avg_imp:.1f}% with a {acc_cost:.2f}% accuracy cost. {rec_text}"
    st.info(f"**Executive Summary:** {exec_summary}")
    
    st.markdown(f"### Mitigation Strategy Applied: `{method}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Before")
        st.markdown(f"**Disparate Impact:** `{before['DI']['value']:.3f}` ({before['DI']['verdict']})")
        st.markdown(f"**Statistical Parity Diff:** `{before['SPD']['value']:.3f}` ({before['SPD']['verdict']})")
        st.markdown(f"**Equal Opportunity Diff:** `{before['EOD']['value']:.3f}` ({before['EOD']['verdict']})")
        st.markdown(f"**Accuracy:** `{baseline_acc*100:.2f}%`")
        
    with col2:
        st.markdown("#### After")
        st.markdown(f"**Disparate Impact:** `{after['DI']['value']:.3f}` ({after['DI']['verdict']}) <span style='color:#10B981'>({di_imp:+.1f}%)</span>", unsafe_allow_html=True)
        st.markdown(f"**Statistical Parity Diff:** `{after['SPD']['value']:.3f}` ({after['SPD']['verdict']}) <span style='color:#10B981'>({spd_imp:+.1f}%)</span>", unsafe_allow_html=True)
        st.markdown(f"**Equal Opportunity Diff:** `{after['EOD']['value']:.3f}` ({after['EOD']['verdict']}) <span style='color:#10B981'>({eod_imp:+.1f}%)</span>", unsafe_allow_html=True)
        st.markdown(f"**Accuracy:** `{fixed_acc*100:.2f}%`")
        
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if method == "Reweighing":
        st.markdown("<p style='font-size: 0.9rem; color: #9CA3AF; margin-bottom: 2rem;'>Note: The reweighed dataset shows similar raw statistics because reweighing adjusts model training weights, not the underlying data distribution. The improvement is realized when you retrain your model using the instance_weight column.</p>", unsafe_allow_html=True)
    
    # Download Section
    st.markdown("### Export & Download")
    st.markdown("<p style='color: #9CA3AF; margin-bottom: 1.5rem;'>Your debiased dataset is ready. Use it to retrain your model for fairer outcomes.</p>", unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns(2)
    
    # Handle CSV data
    df_fixed = st.session_state.fixed_metrics['df_fixed']
    csv_data = df_fixed.to_csv(index=False).encode('utf-8')
    
    if method == "Reweighing":
        csv_label = "📥 Download Reweighed Dataset (CSV)"
        csv_tooltip = "This dataset includes instance weights. Use these weights when training your model to reduce bias."
        csv_file = "fairsight_reweighed_data.csv"
    else:
        csv_label = "📥 Download Fair Predictions Dataset (CSV)"
        csv_tooltip = "This dataset shows original vs bias-corrected predictions for each row."
        csv_file = "fairsight_fair_predictions.csv"

    with col_d1:
        st.download_button(
            label=csv_label,
            data=csv_data,
            file_name=csv_file,
            mime="text/csv",
            help=csv_tooltip,
            type="primary",
            use_container_width=True
        )
        
    with col_d2:
        overview = st.session_state.scan_results['dataset_overview']
        proxy_warnings = st.session_state.shap_results['proxy_warnings'] if "shap_results" in st.session_state else []
        shap_path = st.session_state.shap_results['shap_plot_path'] if "shap_results" in st.session_state else None
        
        with st.spinner("Generating PDF Report..."):
            pdf_bytes = generate_pdf_report(before, after, overview, proxy_warnings, shap_path, baseline_acc, fixed_acc, method, s_col, target_col, exec_summary)
            
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="FairSight_Mitigation_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    # Model download in its own section or below
    _, col_model, _ = st.columns([1, 2, 1])
    with col_model:
        model_bytes = pickle.dumps(model)
        st.download_button(
            label="🤖 Download Fair Model (.pkl)",
            data=model_bytes,
            file_name="fairsight_debiased_model.pkl",
            mime="application/octet-stream",
            type="secondary",
            use_container_width=True
        )
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Start Over with New Dataset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()
