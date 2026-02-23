"""
Feature column mappings for Unified HNC by clinical timeline.

This module defines which columns from the harmonized preprocessed data
belong to each stage of the patient timeline:
- step1 (T0 - Background): Demographics + Exposures
- step2 (T1 - Initial Diagnosis): Anatomy, Clinical staging
- step3 (T2 - Post Surgery): Pathologic staging, Grade, Invasion markers

Columns are matched by exact name or prefix, supporting the one-hot encoded
output of the harmonizer.
"""
from typing import List


# =============================================================================
# STEP1 (T0 - Background): Demographics + Exposures
# =============================================================================

STEP1_EXACT = [
    'sex',
    'age_normalized',
]

STEP1_PREFIX = [
    'smoking_status_',
]


# =============================================================================
# STEP2 (T1 - Initial Diagnosis): Anatomy, Clinical staging
# =============================================================================

STEP2_EXACT = []

STEP2_PREFIX = [
    'primary_site_',
    'has_metastasis_at_diagnosis_',
]


# =============================================================================
# STEP3 (T2 - Post Surgery): Pathologic staging, Grade, Invasion, Margins
# =============================================================================

STEP3_EXACT = [
    'num_positive_lymph_nodes_normalized',
    'num_lymph_nodes_examined_normalized',
    'missingindicator_num_lymph_nodes_examined',
]

STEP3_PREFIX = [
    'pT_stage_',
    'pN_stage_',
    'tumor_grade_',
    'perineural_invasion_',
    'lymphovascular_invasion_',
    'vascular_invasion_',
    'margin_status_',
    'extranodal_extension_',
]


# =============================================================================
# Helper functions
# =============================================================================

def _match_columns(df_columns: List[str], exact: List[str], prefixes: List[str]) -> List[str]:
    """Match columns by exact name or prefix."""
    matched = []
    for col in df_columns:
        if col in exact:
            matched.append(col)
        elif any(col.startswith(prefix) for prefix in prefixes):
            matched.append(col)
    return matched


def get_unified_step1_cols(df_columns: List[str]) -> List[str]:
    """Get columns for step1 (T0 - Background)."""
    return _match_columns(df_columns, STEP1_EXACT, STEP1_PREFIX)


def get_unified_step2_cols(df_columns: List[str]) -> List[str]:
    """Get columns for step2 (T1 - Initial Diagnosis)."""
    return _match_columns(df_columns, STEP2_EXACT, STEP2_PREFIX)


def get_unified_step3_cols(df_columns: List[str]) -> List[str]:
    """Get columns for step3 (T2 - Post Surgery)."""
    return _match_columns(df_columns, STEP3_EXACT, STEP3_PREFIX)


# =============================================================================
# For base dataset (non-graph models): grouped by clinical vs pathological
# =============================================================================

def get_unified_clinical_cols(df_columns: List[str]) -> List[str]:
    """
    Get clinical columns for base dataset (step1 + step2).
    """
    return get_unified_step1_cols(df_columns) + get_unified_step2_cols(df_columns)


def get_unified_pathological_cols(df_columns: List[str]) -> List[str]:
    """
    Get pathological columns for base dataset (step3).
    """
    return get_unified_step3_cols(df_columns)
