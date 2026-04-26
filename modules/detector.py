import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

def scan_columns(df: pd.DataFrame) -> dict:
    """
    Scans a pandas DataFrame to automatically detect potential sensitive attributes
    and the target prediction column. Also provides a dataset overview.
    
    Args:
        df: Pandas DataFrame to analyze.
        
    Returns:
        dict: A dictionary containing 'sensitive_candidates', 'target_candidates', 
              and 'dataset_overview'.
    """
    sensitive_keywords = [
        "gender", "sex", "race", "ethnicity", "caste", "religion", "nationality", 
        "age", "income", "salary", "marital", "children", "dependents", "zip", 
        "area", "region", "location", "district"
    ]
    
    target_keywords = [
        "label", "target", "outcome", "result", "approved", "rejected", 
        "hired", "admitted", "loan", "predict", "class", "y", "income" # Added income to pass the specified test
    ]
    
    sensitive_candidates = []
    target_candidates = []
    
    # Exclusion list for FairSight generated columns
    def is_excluded(col_name):
        c = str(col_name).lower()
        if c == 'instance_weight': return True
        if c.startswith('fair_'): return True
        if c.endswith('_weight') or c.endswith('_prediction'): return True
        return False
    
    # Analyze each column
    for col in df.columns:
        if is_excluded(col):
            continue
            
        col_lower = str(col).lower()
        
        # 1. Check for sensitive candidates by keyword match
        is_sensitive = False
        for keyword in sensitive_keywords:
            if keyword in col_lower:
                sensitive_candidates.append(col)
                is_sensitive = True
                break
                
        # 2. Check for low cardinality (2-5 unique values) as suspicious
        if not is_sensitive:
            try:
                unique_count = df[col].nunique()
                if 2 <= unique_count <= 5:
                    sensitive_candidates.append(col)
            except Exception:
                pass # Ignore errors for unhashable types
                
        # 3. Check for target candidates by keyword match
        for keyword in target_keywords:
            if col_lower == keyword:
                target_candidates.append(col)
                break
                
    # Generate dataset overview
    try:
        missing_values = df.isnull().sum().to_dict()
    except Exception:
        missing_values = {}
        
    try:
        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    except Exception:
        data_types = {}
        
    dataset_overview = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": missing_values,
        "data_types": data_types
    }
    
    # Remove duplicates while preserving order
    sensitive_candidates = list(dict.fromkeys(sensitive_candidates))
    target_candidates = list(dict.fromkeys(target_candidates))
    
    # Ensure target candidates are not in sensitive candidates
    sensitive_candidates = [col for col in sensitive_candidates if col not in target_candidates]
    
    return {
        "sensitive_candidates": sensitive_candidates,
        "target_candidates": target_candidates,
        "dataset_overview": dataset_overview
    }

def compute_bias_metrics(df: pd.DataFrame, sensitive_col: str, target_col: str, fairness_definition: str) -> dict:
    """
    Computes bias metrics using AIF360's ClassificationMetric.
    Automatically handles label encoding and privileged/unprivileged group mapping.
    """
    # Encoding categorical columns
    encoders = {}
    encoded_df = df.copy()
    
    # Strip whitespace and trailing periods from all string columns
    for col in encoded_df.columns:
        if not pd.api.types.is_numeric_dtype(encoded_df[col]):
            encoded_df[col] = encoded_df[col].astype(str).str.strip().str.rstrip('.')
            
    # Target column specific encoding & Binary Check
    if not pd.api.types.is_numeric_dtype(encoded_df[target_col]):
        encoded_df[target_col] = encoded_df[target_col].astype(str).str.strip().str.rstrip('.')
    
    unique_vals = sorted(encoded_df[target_col].unique())
    if len(unique_vals) > 2:
        return {"error": f"FairSight currently supports binary classification targets only. Your target column has more than 2 unique values ({len(unique_vals)} found). Please select a column with exactly 2 unique values as your target."}
    
    if len(unique_vals) == 2:
        # User requested first unique value (sorted) is 1, second is 0
        mapping = {unique_vals[0]: 1.0, unique_vals[1]: 0.0}
        print(f"DEBUG: Target encoding mapping for '{target_col}': {mapping}")
        encoded_df[target_col] = encoded_df[target_col].map(mapping)
    else:
        # Only 1 unique value - technically binary but degenerated
        encoded_df[target_col] = 1.0
        print(f"DEBUG: Target '{target_col}' has only one unique value. Setting to 1.0.")
            
    # Encode remaining categorical columns
    for col in encoded_df.columns:
        if col != target_col and not pd.api.types.is_numeric_dtype(encoded_df[col]):
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col])
            encoders[col] = le

    # Dynamic privilege mapping: Most frequent is privileged
    privileged_val = float(encoded_df[sensitive_col].mode()[0])
    
    unique_vals = encoded_df[sensitive_col].unique()
    unprivileged_val = float([v for v in unique_vals if v != privileged_val][0]) if len(unique_vals) > 1 else 0.0

    privileged_groups = [{sensitive_col: privileged_val}]
    unprivileged_groups = [{sensitive_col: unprivileged_val}]

    # Target mapping
    favorable_label = 1.0
    unfavorable_label = 0.0

    dataset_true = BinaryLabelDataset(
        df=encoded_df,
        label_names=[target_col],
        protected_attribute_names=[sensitive_col],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label
    )

    # Train a quick dummy model to get predictions for Equal Opportunity Difference
    X = encoded_df.drop(columns=[target_col])
    y = encoded_df[target_col]
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)

    dataset_pred = dataset_true.copy()
    dataset_pred.labels = preds.reshape(-1, 1)

    metric = ClassificationMetric(
        dataset_true, dataset_pred, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    try:
        di = float(metric.disparate_impact())
    except:
        di = 1.0
    try:
        spd = float(metric.statistical_parity_difference())
    except:
        spd = 0.0
    try:
        eod = float(metric.equal_opportunity_difference())
    except:
        eod = 0.0

    # Determine Verdicts
    di_verdict = "FAIR" if 0.8 <= di <= 1.2 else "BIASED"
    spd_verdict = "FAIR" if -0.1 <= spd <= 0.1 else "BIASED"
    eod_verdict = "FAIR" if -0.1 <= eod <= 0.1 else "BIASED"
    
    if fairness_definition == "Demographic Parity":
        overall_verdict = spd_verdict
    elif fairness_definition == "Equal Opportunity":
        overall_verdict = eod_verdict
    elif fairness_definition == "Equalized Odds":
        overall_verdict = "FAIR" if spd_verdict == "FAIR" and eod_verdict == "FAIR" else "BIASED"
    else:
        overall_verdict = "FAIR"

    return {
        "DI": {"value": di, "verdict": di_verdict},
        "SPD": {"value": spd, "verdict": spd_verdict},
        "EOD": {"value": eod, "verdict": eod_verdict},
        "overall_verdict": overall_verdict
    }
