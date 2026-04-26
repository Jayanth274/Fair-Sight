import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('Agg') # Ensure headless plotting doesn't break Streamlit
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def explain_bias(df: pd.DataFrame, sensitive_col: str, target_col: str) -> dict:
    """
    Trains a model, runs SHAP explainability, saves a summary plot,
    and detects potential proxy variables for the sensitive attribute.
    """
    # 1. Preprocess data (label encode all strings)
    encoded_df = df.copy()
    
    for col in encoded_df.columns:
        if not pd.api.types.is_numeric_dtype(encoded_df[col]):
            # Strip whitespace and trailing periods to match detector's logic
            encoded_df[col] = encoded_df[col].astype(str).str.strip().str.rstrip('.')
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col])

    # Ensure target_col exists (edge case handling)
    if target_col not in encoded_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    # Exclusion list for FairSight generated columns (must not be used as features)
    generated_cols = [col for col in encoded_df.columns if 
                      col.lower() == 'instance_weight' or 
                      col.lower().endswith('_weight') or 
                      col.lower().endswith('_prediction') or 
                      col.lower().endswith('_fair')]
    
    # We drop them if they are not the target column itself
    cols_to_drop = [c for c in generated_cols if c != target_col]
    if cols_to_drop:
        encoded_df = encoded_df.drop(columns=cols_to_drop)

    # Split features and target
    X = encoded_df.drop(columns=[target_col])
    y = encoded_df[target_col]
    
    # 2. Train simple RandomForest
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    
    # 3. Run SHAP TreeExplainer
    # Use a sample if > 1000 rows to keep the UI highly responsive
    X_sample = X.sample(min(len(X), 1000), random_state=42)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    
    # Handle SHAP output structure based on sklearn version/binary classification
    if isinstance(shap_values, list):
        # RandomForest binary classifier outputs a list: [negative_class, positive_class]
        shap_values_to_plot = shap_values[1]
    else:
        # In some versions, it's a 3D array or just single array
        if len(shap_values.shape) == 3:
            shap_values_to_plot = shap_values[:, :, 1]
        else:
            shap_values_to_plot = shap_values

    # Generate SHAP summary plot
    plt.figure(figsize=(10, 6))
    
    # Apply white text colors for dark background
    plt.rcParams.update({
        "text.color": "#F9FAFB",
        "axes.labelcolor": "#F9FAFB",
        "xtick.color": "#F9FAFB",
        "ytick.color": "#F9FAFB",
        "axes.edgecolor": "#F9FAFB"
    })
    
    shap.summary_plot(shap_values_to_plot, X_sample, show=False)
    
    # Ensure assets directory exists for saving the plot
    os.makedirs('assets', exist_ok=True)
    # Sanitize sensitive_col for filename just in case
    safe_sensitive_col = "".join([c if c.isalnum() else "_" for c in sensitive_col])
    shap_plot_path = f"assets/shap_summary_{safe_sensitive_col}.png"
    
    plt.tight_layout()
    plt.savefig(shap_plot_path, bbox_inches='tight', dpi=300, facecolor='#0E1117', edgecolor='none')
    
    # The user wants dark theme! Matplotlib plots have white backgrounds by default.
    # We should stylize the plot to match the dark theme `#0E1117` or let it be. 
    # Actually, shap.summary_plot overrides a lot, but facecolor helps the edges.
    plt.close()
    
    # 4. Identify top 5 features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values_to_plot).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)
    
    top_5_features = feature_importance['feature'].head(5).tolist()
    top_features_list = feature_importance.head(5).to_dict('records')
    
    # 5. PROXY VARIABLE DETECTION
    proxy_warnings = []
    
    # Check if sensitive column is NOT in top 5
    if sensitive_col not in top_5_features and sensitive_col in X.columns:
        for feature in top_5_features:
            # Compute correlation between top feature and sensitive column
            # Using absolute Pearson correlation
            corr = np.abs(encoded_df[feature].corr(encoded_df[sensitive_col]))
            
            if corr > 0.5:
                warning = f"Removing '{sensitive_col}' won't fix bias because '{feature}' is acting as a proxy for it (Correlation: {corr:.2f})."
                proxy_warnings.append(warning)
                
    # 6. Return structured dictionary
    return {
        "shap_plot_path": shap_plot_path,
        "top_features": top_features_list,
        "proxy_warnings": proxy_warnings
    }
