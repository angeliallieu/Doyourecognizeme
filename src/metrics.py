"""
Biometric Metrics: FMR, FNMR, EER
For Facial Verification System Evaluation
"""

import numpy as np
from typing import Tuple, List, Dict


def calculate_fmr(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Calculate False Match Rate (FMR)
    
    FMR = Amount of False Matches / Total Impostor Attempts

    Args:
        y_true: True Labels (0=Impostor, 1=Genuine)
        y_pred: Predicted Similarity Scores [0, 1]
        threshold: Decision Threshold
    
    Returns:
        FMR: False Match Rate [0, 1]
    """
    # Only Impostor Attempts (y_true == 0)
    impostor_mask = (y_true == 0)
    
    if np.sum(impostor_mask) == 0:
        return 0.0
    
    # False Matches: Impostor Score >= Threshold
    false_matches = np.sum(y_pred[impostor_mask] >= threshold)
    
    # FMR
    fmr = false_matches / np.sum(impostor_mask)
    
    return float(fmr)


def calculate_fnmr(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Calculate False Non-Match Rate (FNMR)
    
    FNMR = Amount False Non-Matches / Total Genuine Attempts
    
    False Non-Match = Genuine is rejected (wrongfully as different classification)
    
    Args:
        y_true: True Labels (0=Impostor, 1=Genuine)
        y_pred: Predicted Similarity Scores [0, 1]
        threshold: Decision Threshold
    
    Returns:
        FNMR: False Non-Match Rate [0, 1]
    """
    # only Genuine Attempts (y_true == 1)
    genuine_mask = (y_true == 1)
    
    if np.sum(genuine_mask) == 0:
        return 0.0
    
    # False Non-Matches: Genuine Score < Threshold
    false_non_matches = np.sum(y_pred[genuine_mask] < threshold)
    
    # FNMR
    fnmr = false_non_matches / np.sum(genuine_mask)
    
    return float(fnmr)


def calculate_eer(y_true: np.ndarray, y_pred: np.ndarray, 
                   num_thresholds: int = 100) -> Tuple[float, float]:
    """
    Calculate Equal Error Rate (EER) and optimum Threshold
    
    EER = Threshold where FMR ≈ FNMR (balance between False Positives & False Negatives)
    
    Args:
        y_true: True Labels (0=Impostor, 1=Genuine)
        y_pred: Predicted Similarity Scores [0, 1]
        num_thresholds: Amount Thresholds to test
    
    Returns:
        Tuple[eer_value, optimal_threshold]
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    
    min_diff = float('inf')
    best_threshold = 0.5
    eer_value = 0.0
    
    for threshold in thresholds:
        fmr = calculate_fmr(y_true, y_pred, threshold)
        fnmr = calculate_fnmr(y_true, y_pred, threshold)
        diff = abs(fmr - fnmr)
        
        if diff < min_diff:
            min_diff = diff
            best_threshold = threshold
            eer_value = (fmr + fnmr) / 2  # EER ≈ FMR ≈ FNMR
    
    return float(eer_value), float(best_threshold)


def calculate_metrics_for_thresholds(y_true: np.ndarray, y_pred: np.ndarray,
                                      thresholds: List[float]) -> List[Dict]:
    """
    Calculate FMR & FNMR for one List of Thresholds
    
    Args:
        y_true: True Labels (0=Impostor, 1=Genuine)
        y_pred: Predicted Similarity Scores [0, 1]
        thresholds: List of Thresholds to test
    
    Returns:
        List[Dict]: List with Metrics per Threshold
            [{'threshold': 0.1, 'fmr': 0.8, 'fnmr': 0.0, 'far': 0.8, 'frr': 0.0}, ...]
    """
    results = []
    
    for threshold in thresholds:
        fmr = calculate_fmr(y_true, y_pred, threshold)
        fnmr = calculate_fnmr(y_true, y_pred, threshold)
        
        results.append({
            'threshold': float(threshold),
            'fmr': fmr,          # False Match Rate (Type I Error)
            'fnmr': fnmr,        # False Non-Match Rate (Type II Error)
            'far': fmr,          # False Acceptance Rate
            'frr': fnmr,         # False Rejection Rate
        })
    
    return results


def calculate_roc_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          num_thresholds: int = 100) -> Tuple[List[float], List[float]]:
    """
    Calculate FPR & TPR für ROC Curve
    
    Args:
        y_true: True Labels (0=Impostor, 1=Genuine)
        y_pred: Predicted Similarity Scores [0, 1]
        num_thresholds: Amount Thresholds to test
    
    Returns:
        Tuple[fpr_list, tpr_list]: for ROC Plotting
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    
    fpr_list = []
    tpr_list = []
    
    for threshold in thresholds:
        # FPR = False Positives / (False Positives + True Negatives)
        # FMR = False Positives / Total Impostors
        fmr = calculate_fmr(y_true, y_pred, threshold)
        
        # TPR = True Positives / (True Positives + False Negatives)
        # 1 - FNMR = True Positives / Total Genuine
        genuine_mask = (y_true == 1)
        if np.sum(genuine_mask) > 0:
            true_positives = np.sum(y_pred[genuine_mask] >= threshold)
            tpr = true_positives / np.sum(genuine_mask)
        else:
            tpr = 0.0
        
        fpr_list.append(fmr)
        tpr_list.append(tpr)
    
    return fpr_list, tpr_list


def print_metrics_table(results: List[Dict], top_thresholds: List[float] = None):
    """
    Print Table with FMR/FNMR Metrics
    
    Args:
        results: Output from calculate_metrics_for_thresholds()
        top_thresholds: Thresholds to highlight (e.g., [0.5, 0.8])
    """
    print("\n" + "="*80)
    print("FMR & FNMR METRICS TABLE")
    print("="*80)
    print(f"\n{'Threshold':<12} {'FMR':<12} {'FNMR':<12} {'Notes':<30}")
    print("-"*80)
    
    for result in results:
        threshold = result['threshold']
        fmr = result['fmr']
        fnmr = result['fnmr']
        
        notes = ""
        if top_thresholds and threshold in top_thresholds:
            notes = "← Current/Tested"
        
        print(f"{threshold:<12.2f} {fmr:<12.4f} {fnmr:<12.4f} {notes:<30}")
    
    print("-"*80)
    print("="*80 + "\n")
