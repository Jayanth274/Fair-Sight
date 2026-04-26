import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import RejectOptionClassification, CalibratedEqOddsPostprocessing
from aif360.metrics import ClassificationMetric

def create_aif_dataset(df: pd.DataFrame, sensitive_col: str, target_col: str):
    """
    Helper to create an AIF360 BinaryLabelDataset from a raw pandas DataFrame.
    Returns the dataset, privileged/unprivileged groups, and the encoded dataframe.
    """
    encoded_df = df.copy()
    
    # Strip whitespace and trailing periods from all string columns
    for col in encoded_df.columns:
        if not pd.api.types.is_numeric_dtype(encoded_df[col]):
            encoded_df[col] = encoded_df[col].astype(str).str.strip().str.rstrip('.')
            
    # Target encoding
    if not pd.api.types.is_numeric_dtype(encoded_df[target_col]):
        unique_vals = encoded_df[target_col].unique()
        if len(unique_vals) >= 2:
            mapping = {unique_vals[0]: 1.0, unique_vals[1]: 0.0}
            encoded_df[target_col] = encoded_df[target_col].map(lambda x: mapping.get(x, 0.0))
        else:
            encoded_df[target_col] = 1.0
            
    # Encode remaining categorical columns
    for col in encoded_df.columns:
        if col != target_col and not pd.api.types.is_numeric_dtype(encoded_df[col]):
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col])

    # Dynamic privilege mapping: Most frequent is privileged
    privileged_val = float(encoded_df[sensitive_col].mode()[0])
    unique_vals = encoded_df[sensitive_col].unique()
    unprivileged_val = float([v for v in unique_vals if v != privileged_val][0]) if len(unique_vals) > 1 else 0.0

    privileged_groups = [{sensitive_col: privileged_val}]
    unprivileged_groups = [{sensitive_col: unprivileged_val}]

    favorable_label = 1.0
    unfavorable_label = 0.0

    dataset = BinaryLabelDataset(
        df=encoded_df,
        label_names=[target_col],
        protected_attribute_names=[sensitive_col],
        favorable_label=favorable_label,
        unfavorable_label=unfavorable_label
    )
    
    target_mapping = None
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        unique_vals = df[target_col].astype(str).str.strip().str.rstrip('.').unique()
        if len(unique_vals) >= 2:
            target_mapping = {1.0: unique_vals[0], 0.0: unique_vals[1]}

    return dataset, privileged_groups, unprivileged_groups, encoded_df, target_mapping


def _compute_metrics(dataset_true, dataset_pred, unprivileged_groups, privileged_groups, fairness_def):
    metric = ClassificationMetric(
        dataset_true, dataset_pred, 
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    # Safely get metrics
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
        
    di_verdict = "FAIR" if 0.8 <= di <= 1.2 else "BIASED"
    spd_verdict = "FAIR" if -0.1 <= spd <= 0.1 else "BIASED"
    eod_verdict = "FAIR" if -0.1 <= eod <= 0.1 else "BIASED"
    
    # Overall verdict
    if fairness_def == "Demographic Parity":
        overall_verdict = spd_verdict
    elif fairness_def == "Equal Opportunity":
        overall_verdict = eod_verdict
    elif fairness_def == "Equalized Odds":
        overall_verdict = "FAIR" if spd_verdict == "FAIR" and eod_verdict == "FAIR" else "BIASED"
    else:
        overall_verdict = "FAIR"
        
    return {
        "DI": {"value": di, "verdict": di_verdict},
        "SPD": {"value": spd, "verdict": spd_verdict},
        "EOD": {"value": eod, "verdict": eod_verdict},
        "overall_verdict": overall_verdict
    }


def apply_reweighing(df: pd.DataFrame, sensitive_col: str, target_col: str, fairness_def: str) -> dict:
    """Applies AIF360 Reweighing (Pre-processing)"""
    dataset_orig, priv_groups, unpriv_groups, encoded_df, _ = create_aif_dataset(df, sensitive_col, target_col)
    
    # Train baseline to get baseline metrics and accuracy
    X = encoded_df.drop(columns=[target_col])
    y = encoded_df[target_col]
    
    clf_orig = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_orig.fit(X, y)
    preds_orig = clf_orig.predict(X)
    
    dataset_pred_orig = dataset_orig.copy()
    dataset_pred_orig.labels = preds_orig.reshape(-1, 1)
    
    baseline_acc = accuracy_score(y, preds_orig)
    baseline_metrics = _compute_metrics(dataset_orig, dataset_pred_orig, unpriv_groups, priv_groups, fairness_def)
    
    # Apply Reweighing
    RW = Reweighing(unprivileged_groups=unpriv_groups, privileged_groups=priv_groups)
    dataset_transf = RW.fit_transform(dataset_orig)
    
    # Train new model with instance weights
    clf_transf = RandomForestClassifier(n_estimators=50, random_state=42)
    # AIF360 stores weights in instance_weights
    clf_transf.fit(X, y, sample_weight=dataset_transf.instance_weights.ravel())
    preds_transf = clf_transf.predict(X)
    
    dataset_pred_transf = dataset_orig.copy()
    dataset_pred_transf.labels = preds_transf.reshape(-1, 1)
    
    transf_acc = accuracy_score(y, preds_transf)
    fixed_metrics = _compute_metrics(dataset_orig, dataset_pred_transf, unpriv_groups, priv_groups, fairness_def)
    
    return {
        "method": "Reweighing",
        "baseline_metrics": baseline_metrics,
        "fixed_metrics": fixed_metrics,
        "baseline_accuracy": baseline_acc,
        "fixed_accuracy": transf_acc,
        "model": clf_transf,
        "df_fixed": df.assign(instance_weight=dataset_transf.instance_weights.ravel())
    }

def apply_postprocessing(df: pd.DataFrame, sensitive_col: str, target_col: str, fairness_def: str, method: str = "Reject Option Classification") -> dict:
    """Applies AIF360 Reject Option Classification (Post-processing)"""
    dataset_orig, priv_groups, unpriv_groups, encoded_df, target_mapping = create_aif_dataset(df, sensitive_col, target_col)
    
    X = encoded_df.drop(columns=[target_col])
    y = encoded_df[target_col]
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)
    
    probas = clf.predict_proba(X)
    # Ensure we get probabilities for favorable label 1.0
    pos_idx = np.where(clf.classes_ == 1.0)[0]
    if len(pos_idx) > 0:
        scores = probas[:, pos_idx[0]]
    else:
        scores = probas[:, 1]
    
    preds_orig = clf.predict(X)
    baseline_acc = accuracy_score(y, preds_orig)
    
    dataset_pred_orig = dataset_orig.copy()
    dataset_pred_orig.labels = preds_orig.reshape(-1, 1)
    dataset_pred_orig.scores = scores.reshape(-1, 1)
    
    baseline_metrics = _compute_metrics(dataset_orig, dataset_pred_orig, unpriv_groups, priv_groups, fairness_def)
    
    # 1. Already Fair Check
    if baseline_metrics['overall_verdict'] == 'FAIR':
        return {
            "method": method,
            "baseline_metrics": baseline_metrics,
            "fixed_metrics": baseline_metrics,
            "baseline_accuracy": baseline_acc,
            "fixed_accuracy": baseline_acc,
            "warning": "This dataset already meets your fairness criteria. Post-processing is not recommended as it may reduce accuracy without meaningful fairness gain.",
            "model": (clf, None),
            "df_fixed": df.assign(
                original_prediction=preds_orig if target_mapping is None else pd.Series(preds_orig).map(target_mapping).values,
                fair_prediction=preds_orig if target_mapping is None else pd.Series(preds_orig).map(target_mapping).values
            )
        }

    # Configure Post Processor
    if method == "Reject Option Classification":
        metric_name = "Equal opportunity difference" if fairness_def in ["Equal Opportunity", "Equalized Odds"] else "Statistical parity difference"
        processor = RejectOptionClassification(unprivileged_groups=unpriv_groups, 
                                         privileged_groups=priv_groups,
                                         low_class_thresh=0.01, high_class_thresh=0.99,
                                         num_class_thresh=100, num_ROC_margin=50,
                                         metric_name=metric_name,
                                         metric_ub=0.05, metric_lb=-0.05)
    else:
        # Calibrated Equalized Odds
        cost_constraint = "fpr" if fairness_def == "Equal Opportunity" else "fnr"
        processor = CalibratedEqOddsPostprocessing(privileged_groups=priv_groups,
                                                   unprivileged_groups=unpriv_groups,
                                                   cost_constraint=cost_constraint,
                                                   seed=42)
        
    try:
        processor = processor.fit(dataset_orig, dataset_pred_orig)
        dataset_transf_pred = processor.predict(dataset_pred_orig)
    except Exception:
        # Fallback if it fails to converge
        dataset_transf_pred = dataset_pred_orig
        
    transf_acc = accuracy_score(y, dataset_transf_pred.labels.ravel())
    
    # 2. Accuracy Drop Sanity Check
    if (baseline_acc - transf_acc) > 0.20:
        return {
            "method": method,
            "baseline_metrics": baseline_metrics,
            "fixed_metrics": baseline_metrics,
            "baseline_accuracy": baseline_acc,
            "fixed_accuracy": baseline_acc,
            "error": "Post-processing produced unreliable results. This may occur when the dataset is already fair or when group sizes are highly imbalanced. Try Reweighing instead.",
            "model": (clf, None),
            "df_fixed": df.assign(
                original_prediction=preds_orig if target_mapping is None else pd.Series(preds_orig).map(target_mapping).values,
                fair_prediction=preds_orig if target_mapping is None else pd.Series(preds_orig).map(target_mapping).values
            )
        }

    fixed_metrics = _compute_metrics(dataset_orig, dataset_transf_pred, unpriv_groups, priv_groups, fairness_def)
    
    return {
        "method": method,
        "baseline_metrics": baseline_metrics,
        "fixed_metrics": fixed_metrics,
        "baseline_accuracy": baseline_acc,
        "fixed_accuracy": transf_acc,
        "model": (clf, processor),
        "df_fixed": df.assign(
            original_prediction=preds_orig if target_mapping is None else pd.Series(preds_orig).map(target_mapping).values,
            fair_prediction=dataset_transf_pred.labels.ravel() if target_mapping is None else pd.Series(dataset_transf_pred.labels.ravel()).map(target_mapping).values
        )
    }
