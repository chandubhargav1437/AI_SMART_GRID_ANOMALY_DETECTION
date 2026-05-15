"""
Phase 6 — Pipeline Tests
4 pytest tests that verify the full pipeline works correctly.
"""

import pytest
import pandas as pd
import sys
sys.path.append('.')

from engine.response import get_response


def test_simulation_output_shape():
    df = pd.read_csv('data/grid_data.csv')
    assert df.shape[0] == 50000, f"Expected 50000 rows, got {df.shape[0]}"
    assert 'MMXU1.PhV.phsA' in df.columns
    assert 'fault_type' in df.columns


def test_iec_columns_present():
    df = pd.read_csv('data/grid_data.csv')
    required = ['MMXU1.PhV.phsA', 'MMXU1.PhV.phsB', 'MMXU1.A.phsA', 'MMXU1.Hz', 'MMXU1.W']
    for col in required:
        assert col in df.columns, f"Missing IEC 61850 column: {col}"


def test_anomaly_prediction_values():
    df = pd.read_csv('data/grid_data_detected.csv')
    assert 'is_anomaly' in df.columns
    unique_vals = set(df['is_anomaly'].unique())
    assert unique_vals.issubset({-1, 1}), f"Unexpected prediction values: {unique_vals}"


def test_response_low_confidence():
    result = get_response(fault_type=1, confidence=0.4)
    assert "Human review" in result

    result2 = get_response(fault_type=5, confidence=0.9)
    assert "Isolate" in result2 or "reroute" in result2.lower()
