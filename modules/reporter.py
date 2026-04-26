import os
import tempfile
from fpdf import FPDF

def generate_pdf_report(before_metrics, after_metrics, dataset_overview, proxy_warnings, shap_plot_path, baseline_acc, fixed_acc, method, s_col, target_col, exec_summary):
    pdf = FPDF()
    pdf.add_page()
    
    # Colors
    color_green = (16, 185, 129)
    color_red = (239, 68, 68)
    color_gray = (107, 114, 128)
    color_black = (0, 0, 0)
    
    # Header
    pdf.set_font("Helvetica", style="B", size=24)
    pdf.set_text_color(124, 58, 237) # Purple
    pdf.cell(w=0, h=15, text="FairSight", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.set_text_color(*color_black)
    pdf.cell(w=0, h=10, text="Bias Mitigation Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(5)
    
    # Executive Summary
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(w=0, h=10, text="Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(w=0, h=8, text=exec_summary, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Overview
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(w=0, h=10, text="Dataset Overview", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(w=0, h=8, text=f"Total Rows: {dataset_overview['row_count']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(w=0, h=8, text=f"Target Column: {target_col}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(w=0, h=8, text=f"Sensitive Attribute: {s_col}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(w=0, h=8, text=f"Mitigation Strategy: {method}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # Metrics Table
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(w=0, h=10, text="Bias Metrics Comparison", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", style="B", size=11)
    
    # Table Header
    col_widths = [60, 60, 60]
    pdf.cell(w=col_widths[0], h=10, text="Metric", border=1, align="C")
    pdf.cell(w=col_widths[1], h=10, text="Before Mitigation", border=1, align="C")
    pdf.cell(w=col_widths[2], h=10, text="After Mitigation", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", size=11)
    
    def write_metric_row(metric_name, b_val, b_verd, a_val, a_verd):
        pdf.set_text_color(*color_black)
        pdf.cell(w=col_widths[0], h=10, text=metric_name, border=1)
        
        pdf.set_text_color(*color_green if b_verd == "FAIR" else color_red)
        pdf.cell(w=col_widths[1], h=10, text=f"{b_val:.3f} ({b_verd})", border=1, align="C")
        
        pdf.set_text_color(*color_green if a_verd == "FAIR" else color_red)
        pdf.cell(w=col_widths[2], h=10, text=f"{a_val:.3f} ({a_verd})", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
        
    write_metric_row("Disparate Impact", before_metrics['DI']['value'], before_metrics['DI']['verdict'], after_metrics['DI']['value'], after_metrics['DI']['verdict'])
    write_metric_row("Statistical Parity Diff", before_metrics['SPD']['value'], before_metrics['SPD']['verdict'], after_metrics['SPD']['value'], after_metrics['SPD']['verdict'])
    write_metric_row("Equal Opportunity Diff", before_metrics['EOD']['value'], before_metrics['EOD']['verdict'], after_metrics['EOD']['value'], after_metrics['EOD']['verdict'])
    
    # Overall Verdict
    pdf.set_text_color(*color_black)
    pdf.cell(w=col_widths[0], h=10, text="OVERALL VERDICT", border=1)
    
    pdf.set_text_color(*color_green if before_metrics['overall_verdict'] == "FAIR" else color_red)
    pdf.cell(w=col_widths[1], h=10, text=before_metrics['overall_verdict'], border=1, align="C")
    
    pdf.set_text_color(*color_green if after_metrics['overall_verdict'] == "FAIR" else color_red)
    pdf.cell(w=col_widths[2], h=10, text=after_metrics['overall_verdict'], border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_text_color(*color_black)
    pdf.ln(5)
    
    # Accuracy Cost
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(w=0, h=10, text="Accuracy & Fairness Tradeoff", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(w=0, h=8, text=f"Baseline Accuracy: {baseline_acc*100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(w=0, h=8, text=f"Mitigated Accuracy: {fixed_acc*100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    cost = (baseline_acc - fixed_acc) * 100
    pdf.set_text_color(*color_red if cost > 5 else color_black)
    pdf.cell(w=0, h=8, text=f"Accuracy Cost: {cost:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(*color_black)
    pdf.ln(5)
    
    # Proxy Warnings
    if proxy_warnings:
        pdf.set_text_color(*color_red)
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(w=0, h=10, text="Proxy Variable Warnings", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        for w in proxy_warnings:
            pdf.multi_cell(w=0, h=8, text=f"- {w}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*color_black)
        pdf.ln(5)
        
    # SHAP Plot
    if shap_plot_path and os.path.exists(shap_plot_path):
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=14)
        pdf.cell(w=0, h=10, text="SHAP Explainability Summary", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.image(shap_plot_path, x=10, w=190)

    # Save to temp file and return bytes
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, "FairSight_Report.pdf")
    pdf.output(filepath)
    
    with open(filepath, "rb") as f:
        pdf_bytes = f.read()
        
    return pdf_bytes

def generate_fairness_certificate(metadata, metrics):
    """Generates a professional, compliance-style PDF certificate for fair datasets."""
    from datetime import date
    pdf = FPDF()
    pdf.add_page()
    
    # Colors
    color_green = (16, 185, 129)
    color_black = (0, 0, 0)
    color_light_gray = (209, 213, 219)
    
    # 1. Header Bar
    pdf.set_fill_color(*color_green)
    pdf.rect(x=0, y=0, w=210, h=40, style="F")
    
    pdf.set_xy(0, 10)
    pdf.set_font("Helvetica", style="B", size=28)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(w=0, h=15, text="FairSight", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", size=14)
    pdf.cell(w=0, h=10, text="Fairness Certificate", align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_xy(10, 50)
    pdf.set_text_color(*color_black)
    
    # 2. Dataset Details
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(w=0, h=10, text="Dataset Details", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_draw_color(*color_light_gray)
    pdf.line(x1=10, y1=pdf.get_y(), x2=200, y2=pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=11)
    # Detail rows
    def add_detail(label, value):
        pdf.set_font("Helvetica", style="B", size=11)
        pdf.cell(w=50, h=8, text=f"{label}:")
        pdf.set_font("Helvetica", size=11)
        pdf.cell(w=0, h=8, text=str(value), new_x="LMARGIN", new_y="NEXT")
        
    add_detail("Dataset Name", metadata['filename'])
    add_detail("Analysis Date", date.today().strftime("%B %d, %Y"))
    add_detail("Sensitive Attribute", ", ".join(metrics.keys()))
    pdf.ln(10)
    
    # 3. Fairness Analysis Summary Table
    pdf.set_font("Helvetica", style="B", size=14)
    pdf.cell(w=0, h=10, text="Fairness Analysis Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.line(x1=10, y1=pdf.get_y(), x2=200, y2=pdf.get_y())
    pdf.ln(5)
    
    # Table Header
    col_widths = [50, 40, 50, 50]
    pdf.set_font("Helvetica", style="B", size=11)
    pdf.set_fill_color(243, 244, 246)
    pdf.cell(w=col_widths[0], h=10, text="Metric", border=1, align="C", fill=True)
    pdf.cell(w=col_widths[1], h=10, text="Value", border=1, align="C", fill=True)
    pdf.cell(w=col_widths[2], h=10, text="Status", border=1, align="C", fill=True)
    pdf.cell(w=col_widths[3], h=10, text="Fair Range", border=1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", size=11)
    ranges = {"DI": "0.8 - 1.2", "SPD": "-0.1 - 0.1", "EOD": "-0.1 - 0.1"}
    labels = {"DI": "Disparate Impact", "SPD": "Stat. Parity Diff.", "EOD": "Equal Opp. Diff."}
    
    for s_col, m in metrics.items():
        for m_key in ['DI', 'SPD', 'EOD']:
            val = m[m_key]['value']
            verd = m[m_key]['verdict']
            
            pdf.cell(w=col_widths[0], h=10, text=labels[m_key], border=1)
            pdf.cell(w=col_widths[1], h=10, text=f"{val:.3f}", border=1, align="C")
            
            # Status with color coding
            if verd == "FAIR":
                pdf.set_text_color(*color_green)
            else:
                pdf.set_text_color(239, 68, 68)
            pdf.cell(w=col_widths[2], h=10, text=verd, border=1, align="C")
            
            pdf.set_text_color(*color_black)
            pdf.cell(w=col_widths[3], h=10, text=ranges[m_key], border=1, align="C", new_x="LMARGIN", new_y="NEXT")
            
    pdf.ln(20)
    
    # 4. Verdict Stamp
    pdf.set_draw_color(*color_green)
    pdf.set_line_width(1)
    pdf.set_font("Helvetica", style="B", size=32)
    pdf.set_text_color(*color_green)
    pdf.cell(w=0, h=30, text="CERTIFIED FAIR \xbb", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_line_width(0.2) # Reset
    
    # 5. Footer & Disclaimer
    pdf.set_y(-40)
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(107, 114, 128)
    footer_text = f"Generated by FairSight Bias Detection Platform | Powered by AIF360 & SHAP | Date: {date.today().strftime('%Y-%m-%d')}"
    pdf.cell(w=0, h=10, text=footer_text, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", style="I", size=8)
    disclaimer = "This certificate reflects fairness analysis based on the selected fairness definition. Results may vary with different fairness criteria."
    pdf.multi_cell(w=0, h=5, text=disclaimer, align="C")
    
    # Save to temp file and return bytes
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, "FairSight_Certificate.pdf")
    pdf.output(filepath)
    
    with open(filepath, "rb") as f:
        pdf_bytes = f.read()
        
    return pdf_bytes
