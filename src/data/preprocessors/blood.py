import os
import json
import itertools
import pandas as pd
from typing import Tuple, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

pd.set_option('future.no_silent_downcasting', True)


def load_blood_data(data_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load blood data and reference ranges."""
    blood_path = os.path.join(data_root, "StructuredData", "blood_data.json")
    range_path = os.path.join(data_root, "StructuredData", "blood_data_reference_ranges.json")
    
    with open(blood_path, 'r') as file:
        blood_data = json.load(file)
    df = pd.DataFrame(blood_data)
    
    with open(range_path, 'r') as file:
        range_data = json.load(file)
    df_range = pd.DataFrame(range_data)
    
    return df, df_range

# ============================================================
# Feature Engineering + Most Preprocessing Functions
# ============================================================

def fill_missing_patients(df: pd.DataFrame, data_clinical: pd.DataFrame) -> pd.DataFrame:
    # Mark existing rows before cartesian product
    df['row_existed'] = True

    analytes_name = df.analyte_name.unique()
    
    all_combinations = pd.DataFrame(
        itertools.product(data_clinical['patient_id'].unique(), analytes_name),
        columns=['patient_id', 'analyte_name']
    )
    
    df_complete = all_combinations.merge(df, on=['patient_id', 'analyte_name'], how='left')
    # Fill new rows (created by cartesian product) with row_existed=False
    df_complete['row_existed'] = df_complete['row_existed'].fillna(False).infer_objects(copy=False).astype(int)
    
    return df_complete


def determine_range(row: pd.Series) -> str:
    """Determine if value is lower/within/upper the normal range"""
    if pd.isna(row['value']):
        return None
    
    if row['sex']:  # Male
        min_val = row['normal_male_min']
        max_val = row['normal_male_max']
    else:  # Female
        min_val = row['normal_female_min']
        max_val = row['normal_female_max']
    
    if pd.isna(min_val) or pd.isna(max_val):
        return None
    
    if row['value'] < min_val:
        return "lower"
    elif row['value'] <= max_val:
        return "within"
    else:
        return "upper"


def add_range_indications(df: pd.DataFrame, df_range: pd.DataFrame, data_clinical: pd.DataFrame) -> pd.DataFrame:
    """Add range category based on sex and reference ranges."""
    df = df.merge(data_clinical[['patient_id', 'sex']], on='patient_id', how='left')
    df = df.merge(df_range, on="analyte_name", how="left")
    df['range'] = df.apply(determine_range, axis=1)
    # No dummy_na=True since we use add_indicator=True in imputer
    df = pd.get_dummies(df, columns=["range"], dummy_na=False, dtype=int)
    return df


def pivot_blood_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to wide format: one row per patient."""
    df = df.pivot_table(
        index="patient_id",
        columns="analyte_name",
        values=["value", "row_existed", "range_lower", "range_upper", "range_within"],
        aggfunc="first"
    )
    
    df.columns = [f"{val}_{col}" for val, col in df.columns]
    df = df.reset_index()
    
    return df


# ============================================================
# Main API
# ============================================================

def prepare_blood_data_features(
    data_root: str, 
    data_clinical: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Feature engineering for blood data
    
    Args:
        data_root: Root directory of HANCOCK dataset
        data_clinical: Clinical dataframe (for patient list and sex info)
        
    Returns:
        Tuple of:
            - df: Wide format DataFrame
            - value_columns: List of column names to transform
    """
    df, df_range = load_blood_data(data_root)
    df = df[["patient_id", "value", "analyte_name"]]
    df = fill_missing_patients(df, data_clinical)
    df = add_range_indications(df, df_range, data_clinical)
    
    # Select columns
    columns = [
        "patient_id", "analyte_name", "value", "row_existed",
        "range_lower", "range_upper", "range_within"
    ]
    df = df[columns]
    # Pivot to wide format
    df = pivot_blood_data(df)
    
    # Identify value columns (these will be transformed)
    value_columns = [col for col in df.columns if col.startswith('value_')]
    
    return df, value_columns


def create_blood_transformer(value_columns: List[str]) -> ColumnTransformer:
    """
    Create sklearn ColumnTransformer for blood data transformation.
    
    Args:
        value_columns: List of column names to apply transformation
    
    Returns:
        ColumnTransformer (not fitted yet)
    """
    # Pipeline: imputation (median) + scaling (min-max) + missing indicator
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', MinMaxScaler())
    ])
    
    # ColumnTransformer: apply pipeline to value columns only
    ct = ColumnTransformer(
        transformers=[
            ('value_cols', numeric_pipeline, value_columns)
        ],
        remainder='passthrough',  # Keep row_existed, range_lower, range_upper, range_within unchanged
        verbose_feature_names_out=False
    )
    
    return ct


def apply_blood_transformer(
    df: pd.DataFrame, 
    transformer: ColumnTransformer, 
    value_columns: List[str]
) -> pd.DataFrame:
    """
    Apply fitted transformer and reconstruct DataFrame.
    
    Args:
        df: Wide format DataFrame (from prepare_blood_data_features)
        transformer: Fitted ColumnTransformer
        value_columns: List of value column names (for reconstruction)
    
    Returns:
        Transformed DataFrame with:
        - value_normalized_* : scaled values
        - missingindicator_* : binary indicators (from add_indicator=True)
        - row_existed_* : binary indicators (original row existed)
        - range_lower/upper/within_* : unchanged
    """
    # Transform
    transformed_array = transformer.transform(df)
    
    # Reconstruct columns
    # [value_1_scaled, value_2_scaled, ..., indicator_1, indicator_2, ...]    
    # Scaled value columns
    scaled_value_cols = [col.replace('value_', 'value_normalized_') for col in value_columns]
    # Indicator columns
    indicator_cols = [col.replace('value_', 'missingindicator_') for col in value_columns]
    # Remainder columns (row_existed, range_lower, range_upper, range_within)
    remainder_columns = [col for col in df.columns if col not in value_columns]
    # All columns in order: scaled values + indicators + remainder
    all_columns = scaled_value_cols + indicator_cols + remainder_columns
    
    df_transformed = pd.DataFrame(
        transformed_array,
        columns=all_columns,
        index=df.index
    )
    
    return df_transformed
