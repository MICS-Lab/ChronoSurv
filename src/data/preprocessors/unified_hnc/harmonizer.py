"""
Unified HNC data harmonizer.

Harmonizes HANCOCK and TCGA-HNSC datasets into a common schema for joint training.

Handles:
- Value mapping (smoking, staging, grading, invasions, margins, etc.)
- Feature encoding (one-hot, binary)
- Missing indicator creation for continuous features
- Transformer creation and application for continuous feature scaling
"""

import json
import os
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# =============================================================================
# Value mapping dictionaries
# =============================================================================

# --- Smoking status ---
HANCOCK_SMOKING_MAP = {
    'non-smoker': 'never',
    'former': 'former',
    'smoker': 'current',
}

TCGA_SMOKING_MAP = {
    'Lifelong Non-Smoker': 'never',
    'Current Smoker': 'current',
    'Current Reformed Smoker for < or = 15 yrs': 'former',
    'Current Reformed Smoker for > 15 yrs': 'former',
    'Current Reformed Smoker, Duration Not Specified': 'former',
    # 'Not Reported' and 'Unknown' → None (will become NaN)
}

# --- Primary site ---
HANCOCK_PRIMARY_SITE_MAP = {
    'Oropharynx': 'oropharynx',
    'Oral_Cavity': 'oral_cavity',
    'Larynx': 'larynx',
    'Hypopharynx': 'hypopharynx',
    'CUP': 'other',
}

TCGA_PRIMARY_SITE_MAP = {
    'Oropharynx': 'oropharynx',
    'Base of tongue': 'oropharynx',
    'Tonsil': 'oropharynx',
    'Floor of mouth': 'oral_cavity',
    'Other and unspecified parts of tongue': 'oral_cavity',
    'Other and unspecified parts of mouth': 'oral_cavity',
    'Gum': 'oral_cavity',
    'Palate': 'oral_cavity',
    'Larynx': 'larynx',
    'Hypopharynx': 'hypopharynx',
    'Lip': 'lip',
    'Other and ill-defined sites in lip, oral cavity and pharynx': 'other',
    'Bones, joints and articular cartilage of other and unspecified sites': 'other',
}

# --- pT stage ---
HANCOCK_PT_MAP = {
    'pT1': 'T1', 'pT1a': 'T1', 'pT1b': 'T1',
    'pT2': 'T2',
    'pT3': 'T3',
    'pT4a': 'T4a',
    'pT4b': 'T4b',
    'pTis': 'other_early',
    'TX': 'TX',
}

TCGA_PT_MAP = {
    'T0': 'other_early',
    'T1': 'T1',
    'T2': 'T2',
    'T3': 'T3',
    'T4': 'T4a',   # T4 without suffix → merge into T4a (most common T4 substage)
    'T4a': 'T4a',
    'T4b': 'T4b',
    'TX': 'TX',
}

# --- pN stage ---
HANCOCK_PN_MAP = {
    'pN0': 'N0',
    'pN1': 'N1', 'pN1a': 'N1',
    'pN2': 'N2',
    'pN2a': 'N2a', 'pN2b': 'N2b', 'pN2c': 'N2c',
    'pN3': 'N3', 'pN3b': 'N3',
    'NX': 'NX',
}

TCGA_PN_MAP = {
    'N0': 'N0',
    'N1': 'N1',
    'N2': 'N2',
    'N2a': 'N2a', 'N2b': 'N2b', 'N2c': 'N2c',
    'N3': 'N3',
    'NX': 'NX',
}

# --- Tumor grade ---
HANCOCK_GRADE_MAP = {
    'G1': 'G1', 'G2': 'G2', 'G3': 'G3',
    'hpv_association_p16': 'unknown',  # data quality issue in HANCOCK
}

TCGA_GRADE_MAP = {
    'G1': 'G1', 'G2': 'G2', 'G3': 'G3', 'G4': 'G4', 'GX': 'unknown',
}

# --- Margin status ---
HANCOCK_MARGIN_MAP = {
    'R0': 'negative',
    'R1': 'positive',
    'R2': 'positive',
    'RX': 'unknown',
}

TCGA_MARGIN_MAP = {
    'Uninvolved': 'negative',
    'Involved': 'positive',
    'Indeterminate': 'unknown',
}

# --- Extranodal extension ---
HANCOCK_EXTRANODAL_MAP = {
    'no': 'no',
    'yes': 'yes',
}

TCGA_EXTRANODAL_MAP = {
    'No Extranodal Extension': 'no',
    'Microscopic Extension': 'yes',
    'Gross Extension': 'yes',
}

# --- Binary yes/no mapping ---
YES_NO_MAP = {'no': 'no', 'yes': 'yes'}
YES_NO_MAP_TCGA = {'No': 'no', 'Yes': 'yes'}


# =============================================================================
# Utility
# =============================================================================

def _safe_map(value, mapping: dict, default=None):
    """Map a value using a dictionary, returning default for NaN/missing/unmapped."""
    if pd.isna(value) or value is None:
        return default
    return mapping.get(str(value), default)


# =============================================================================
# Feature extraction from raw JSON
# =============================================================================

def _extract_hancock_patients(hancock_root: str) -> pd.DataFrame:
    """Extract HANCOCK patients into the unified raw schema."""
    clinical_path = os.path.join(hancock_root, "StructuredData", "clinical_data.json")
    patho_path = os.path.join(hancock_root, "StructuredData", "pathological_data.json")

    with open(clinical_path) as f:
        clinical_raw = json.load(f)
    with open(patho_path) as f:
        patho_raw = json.load(f)

    patho_dict = {p['patient_id']: p for p in patho_raw}

    records = []
    for c in clinical_raw:
        pid = c['patient_id']
        p = patho_dict.get(pid, {})

        record = {
            'patient_id': pid,
            'source': 'hancock',
            # Survival target
            'event': 1 if c.get('survival_status') == 'deceased' else 0,
            'time': c.get('days_to_last_information'),
            # ===== Step 1 - Common =====
            'sex': c.get('sex'),
            'age': c.get('age_at_initial_diagnosis'),
            'smoking_status': _safe_map(c.get('smoking_status'), HANCOCK_SMOKING_MAP),
            # Step 1 - TCGA only (set to 0 / None for HANCOCK)
            'race': None,
            'ethnicity': None,
            'pack_years_smoked': 0.0,
            'alcohol_history': None,
            'alcohol_drinks_per_day': 0.0,
            # ===== Step 2 - Common =====
            'primary_site': _safe_map(p.get('primary_tumor_site'), HANCOCK_PRIMARY_SITE_MAP),
            'has_metastasis_at_diagnosis': _safe_map(
                c.get('primarily_metastasis'), {'no': 'no', 'yes': 'yes'}
            ),
            # Step 2 - TCGA only (None for HANCOCK)
            'tissue_or_organ_of_origin': None,
            'laterality': None,
            'primary_diagnosis': None,
            'morphology': None,
            'prior_malignancy': None,
            'prior_treatment': None,
            'synchronous_malignancy': None,
            'ajcc_clinical_stage': None,
            'ajcc_clinical_t': None,
            'ajcc_clinical_n': None,
            # ===== Step 3 - Common =====
            'pT_stage': _safe_map(p.get('pT_stage'), HANCOCK_PT_MAP),
            'pN_stage': _safe_map(p.get('pN_stage'), HANCOCK_PN_MAP),
            'tumor_grade': _safe_map(p.get('grading'), HANCOCK_GRADE_MAP),
            'perineural_invasion': _safe_map(p.get('perineural_invasion_Pn'), YES_NO_MAP),
            'lymphovascular_invasion': _safe_map(p.get('lymphovascular_invasion_L'), YES_NO_MAP),
            'vascular_invasion': _safe_map(p.get('vascular_invasion_V'), YES_NO_MAP),
            'margin_status': _safe_map(p.get('resection_status'), HANCOCK_MARGIN_MAP),
            'extranodal_extension': _safe_map(p.get('perinodal_invasion'), HANCOCK_EXTRANODAL_MAP),
            'num_positive_lymph_nodes': p.get('number_of_positive_lymph_nodes'),
            'num_lymph_nodes_examined': p.get('number_of_resected_lymph_nodes'),
            'infiltration_depth_mm': p.get('infiltration_depth_in_mm'),
            # Step 3 - HANCOCK only
            'hpv_status': _safe_map(
                p.get('hpv_association_p16'),
                {'positive': 'positive', 'negative': 'negative', 'not_tested': 'unknown'}
            ),
            'histologic_type': p.get('histologic_type'),
            'resection_status_cis': p.get('resection_status_carcinoma_in_situ'),
            'carcinoma_in_situ': p.get('carcinoma_in_situ'),
            'closest_resection_margin_cm': p.get('closest_resection_margin_in_cm'),
            # Step 3 - TCGA only (None for HANCOCK)
            'ajcc_pathologic_stage': None,
            'ajcc_pathologic_m': None,
        }

        # Metastasis locations (multi-hot, step3, HANCOCK only)
        met_locs_str = c.get('metastasis_1_locations')
        possible_met_locs = [
            'Lung', 'Bones', 'LymphNodes', 'Liver', 'SoftTissue',
            'Peritoneum', 'Skin', 'Pleura', 'Brain', 'Adrenal', 'OtherOrgans',
        ]
        for loc in possible_met_locs:
            col_name = f'metastasis_location_{loc.lower()}'
            if pd.notna(met_locs_str) and met_locs_str != '':
                record[col_name] = 1 if loc in str(met_locs_str).split() else 0
            else:
                record[col_name] = 0

        records.append(record)

    return pd.DataFrame(records)


def _get_tcga_primary_diagnosis(patient: dict) -> dict:
    """Return the primary diagnosis for a TCGA patient."""
    for d in patient.get('diagnoses', []):
        if d.get('diagnosis_is_primary_disease') == 'true':
            return d
    diagnoses = patient.get('diagnoses', [])
    return diagnoses[0] if diagnoses else {}


def _extract_tcga_patients(tcga_root: str) -> pd.DataFrame:
    """Extract TCGA-HNSC patients into the unified raw schema."""
    clinical_path = os.path.join(tcga_root, "clinical_data.json")

    with open(clinical_path) as f:
        raw_data = json.load(f)

    records = []
    for patient in raw_data:
        demo = patient.get('demographic', {})
        primary = _get_tcga_primary_diagnosis(patient)

        # Exposure extraction
        tobacco_status, pack_years = None, None
        alcohol_history, alcohol_drinks = None, None
        for exp in patient.get('exposures', []):
            if exp.get('tobacco_smoking_status'):
                tobacco_status = exp.get('tobacco_smoking_status')
                pack_years = exp.get('pack_years_smoked')
            if 'alcohol_history' in exp:
                alcohol_history = exp.get('alcohol_history')
                alcohol_drinks = exp.get('alcohol_drinks_per_day')

        # Pathology details
        path_detail = {}
        for pd_entry in primary.get('pathology_details', []):
            for key in ['margin_status', 'lymph_nodes_tested', 'lymph_nodes_positive',
                        'perineural_invasion_present', 'vascular_invasion_present',
                        'lymphatic_invasion_present', 'extranodal_extension']:
                if pd_entry.get(key) is not None and key not in path_detail:
                    path_detail[key] = pd_entry[key]

        # Survival
        vital_status = demo.get('vital_status')
        if vital_status == 'Dead':
            event = 1
            time = demo.get('days_to_death')
        elif vital_status == 'Alive':
            event = 0
            time = primary.get('days_to_last_follow_up')
        else:
            event = None
            time = None

        record = {
            'patient_id': patient.get('submitter_id'),
            'source': 'tcga',
            'event': event,
            'time': time,
            # ===== Step 1 - Common =====
            'sex': demo.get('gender'),
            'age': demo.get('age_at_index'),
            'smoking_status': _safe_map(tobacco_status, TCGA_SMOKING_MAP),
            # Step 1 - TCGA only
            'race': demo.get('race'),
            'ethnicity': demo.get('ethnicity'),
            'pack_years_smoked': pack_years,  # may be NaN
            'alcohol_history': alcohol_history,
            'alcohol_drinks_per_day': alcohol_drinks,  # may be NaN
            # ===== Step 2 - Common =====
            'primary_site': _safe_map(patient.get('primary_site'), TCGA_PRIMARY_SITE_MAP),
            'has_metastasis_at_diagnosis': _safe_map(
                primary.get('ajcc_clinical_m'), {'M0': 'no', 'M1': 'yes'}
            ),
            # Step 2 - TCGA only
            'tissue_or_organ_of_origin': primary.get('tissue_or_organ_of_origin'),
            'laterality': primary.get('laterality'),
            'primary_diagnosis': primary.get('primary_diagnosis'),
            'morphology': primary.get('morphology'),
            'prior_malignancy': primary.get('prior_malignancy'),
            'prior_treatment': primary.get('prior_treatment'),
            'synchronous_malignancy': primary.get('synchronous_malignancy'),
            'ajcc_clinical_stage': primary.get('ajcc_clinical_stage'),
            'ajcc_clinical_t': primary.get('ajcc_clinical_t'),
            'ajcc_clinical_n': primary.get('ajcc_clinical_n'),
            # ===== Step 3 - Common =====
            'pT_stage': _safe_map(primary.get('ajcc_pathologic_t'), TCGA_PT_MAP),
            'pN_stage': _safe_map(primary.get('ajcc_pathologic_n'), TCGA_PN_MAP),
            'tumor_grade': _safe_map(primary.get('tumor_grade'), TCGA_GRADE_MAP),
            'perineural_invasion': _safe_map(
                path_detail.get('perineural_invasion_present'), YES_NO_MAP_TCGA
            ),
            'lymphovascular_invasion': _safe_map(
                path_detail.get('lymphatic_invasion_present'), YES_NO_MAP_TCGA
            ),
            'vascular_invasion': _safe_map(
                path_detail.get('vascular_invasion_present'), YES_NO_MAP_TCGA
            ),
            'margin_status': _safe_map(
                path_detail.get('margin_status'), TCGA_MARGIN_MAP
            ),
            'extranodal_extension': _safe_map(
                path_detail.get('extranodal_extension'), TCGA_EXTRANODAL_MAP
            ),
            'num_positive_lymph_nodes': path_detail.get('lymph_nodes_positive'),
            'num_lymph_nodes_examined': path_detail.get('lymph_nodes_tested'),
            'infiltration_depth_mm': 0.0,  # Not available in TCGA
            # Step 3 - HANCOCK only (None for TCGA)
            'hpv_status': None,
            'histologic_type': None,
            'resection_status_cis': None,
            'carcinoma_in_situ': None,
            'closest_resection_margin_cm': None,
            # Step 3 - TCGA only
            'ajcc_pathologic_stage': primary.get('ajcc_pathologic_stage'),
            'ajcc_pathologic_m': primary.get('ajcc_pathologic_m'),
        }

        # Metastasis locations (all 0 for TCGA)
        possible_met_locs = [
            'Lung', 'Bones', 'LymphNodes', 'Liver', 'SoftTissue',
            'Peritoneum', 'Skin', 'Pleura', 'Brain', 'Adrenal', 'OtherOrgans',
        ]
        for loc in possible_met_locs:
            record[f'metastasis_location_{loc.lower()}'] = 0

        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# Feature encoding
# =============================================================================

# All categorical columns to one-hot encode
CATEGORICAL_COLUMNS = [
    # Common
    'smoking_status', 'primary_site', 'has_metastasis_at_diagnosis',
    'pT_stage', 'pN_stage', 'tumor_grade',
    'perineural_invasion', 'lymphovascular_invasion', 'vascular_invasion',
    'margin_status', 'extranodal_extension',
    # HANCOCK only
    'hpv_status', 'histologic_type', 'resection_status_cis',
    'carcinoma_in_situ', 'closest_resection_margin_cm',
    # TCGA only
    'race', 'ethnicity', 'alcohol_history',
    'tissue_or_organ_of_origin', 'laterality',
    'primary_diagnosis', 'morphology',
    'prior_malignancy', 'prior_treatment', 'synchronous_malignancy',
    'ajcc_clinical_stage', 'ajcc_clinical_t', 'ajcc_clinical_n',
    'ajcc_pathologic_stage', 'ajcc_pathologic_m',
]


def _encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode features in the unified DataFrame.

    - Binary encoding for sex (1=male, 0=female)
    - One-hot encoding for all categorical columns (with dummy_na=True)
    - Fill cross-dataset NaN in one-hot and multi-hot columns with 0
    """
    df = df.copy()

    # Binary encoding for sex
    df['sex'] = (df['sex'] == 'male').astype(float)

    # One-hot encode categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df = pd.get_dummies(df, columns=[col], dummy_na=True, dtype=int)

    # Fill NaN in one-hot columns with 0 (cross-dataset features)
    ohe_cols = [
        c for c in df.columns
        if any(c.startswith(f'{cat}_') for cat in CATEGORICAL_COLUMNS)
    ]
    df[ohe_cols] = df[ohe_cols].fillna(0)

    # Fill NaN in multi-hot metastasis location columns with 0
    met_cols = [c for c in df.columns if c.startswith('metastasis_location_')]
    if met_cols:
        df[met_cols] = df[met_cols].fillna(0)

    return df


# =============================================================================
# Continuous columns configuration
# =============================================================================

# Columns that need median imputation (may have NaN within or across datasets)
MEDIAN_IMPUTE_COLUMNS = [
    'age',
    'infiltration_depth_mm',
    'pack_years_smoked',
    'alcohol_drinks_per_day',
    'num_lymph_nodes_examined',
]

# Columns with constant=0 imputation (clinical meaning: 0 if not measured)
CONSTANT_IMPUTE_COLUMNS = [
    'num_positive_lymph_nodes',
]

# Columns that should get a missing indicator (NaN = genuinely missing)
INDICATOR_COLUMNS = [
    'infiltration_depth_mm',
    'pack_years_smoked',
    'alcohol_drinks_per_day',
    'num_lymph_nodes_examined',
]

ALL_CONTINUOUS_COLUMNS = MEDIAN_IMPUTE_COLUMNS + CONSTANT_IMPUTE_COLUMNS


# =============================================================================
# Main API
# =============================================================================

def prepare_unified_features(
    hancock_root: str,
    tcga_root: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and harmonize features from HANCOCK and TCGA-HNSC.

    Produces a unified DataFrame with:
    - Common features mapped to shared column names
    - Dataset-specific features (0-filled for the other dataset)
    - One-hot encoded categoricals
    - Missing indicators for continuous columns with potential NaN

    Args:
        hancock_root: Path to HANCOCK data directory
        tcga_root: Path to TCGA-HNSC data directory

    Returns:
        df: Unified DataFrame (encoded, not yet scaled)
        continuous_columns: List of continuous column names to transform
    """
    # Extract from both datasets into unified schema
    df_hancock = _extract_hancock_patients(hancock_root)
    df_tcga = _extract_tcga_patients(tcga_root)

    # Combine
    df = pd.concat([df_hancock, df_tcga], ignore_index=True)

    # Add missing indicators BEFORE encoding (so they survive as binary columns)
    for col in INDICATOR_COLUMNS:
        df[f'missingindicator_{col}'] = df[col].isna().astype(float)

    # Encode categorical features
    df = _encode_features(df)

    return df, ALL_CONTINUOUS_COLUMNS


def create_unified_transformer() -> ColumnTransformer:
    """
    Create sklearn ColumnTransformer for unified data.

    Groups:
    - Median imputation + MinMax scaling: age, infiltration_depth_mm,
      pack_years_smoked, alcohol_drinks_per_day, num_lymph_nodes_examined
    - Constant=0 imputation + MinMax scaling: num_positive_lymph_nodes

    Returns:
        ColumnTransformer (not fitted yet)
    """
    median_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    constant_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', MinMaxScaler())
    ])

    ct = ColumnTransformer(
        transformers=[
            ('median', median_pipeline, MEDIAN_IMPUTE_COLUMNS),
            ('constant', constant_pipeline, CONSTANT_IMPUTE_COLUMNS),
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    )

    return ct


def apply_unified_transformer(
    df: pd.DataFrame,
    transformer: ColumnTransformer,
) -> pd.DataFrame:
    """
    Apply fitted transformer and reconstruct DataFrame with clean column names.

    Args:
        df: DataFrame from prepare_unified_features
        transformer: Fitted ColumnTransformer

    Returns:
        Transformed DataFrame with *_normalized continuous columns
    """
    transformed_array = transformer.transform(df)

    # Reconstruct column names
    # Order: [median_imputed_cols, constant_imputed_cols, remainder_cols]
    scaled_names = (
        [f'{col}_normalized' for col in MEDIAN_IMPUTE_COLUMNS] +
        [f'{col}_normalized' for col in CONSTANT_IMPUTE_COLUMNS]
    )
    remainder_columns = [
        col for col in df.columns
        if col not in ALL_CONTINUOUS_COLUMNS
    ]
    all_columns = scaled_names + remainder_columns

    df_transformed = pd.DataFrame(
        transformed_array,
        columns=all_columns,
        index=df.index,
    )

    return df_transformed

