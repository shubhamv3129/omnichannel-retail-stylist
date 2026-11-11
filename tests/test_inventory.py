"""
Unit tests for inventory.py.
Validates ERP/inventory checks, stock level reports, and deterministic stockout behavior.
"""

import json
from unittest.mock import patch

import inventory


def test_check_local_inventory_demo_stockout():
    """Verify deterministic stockout returns zero inventory across all locations."""
    result_raw = inventory.check_local_inventory("14325")
    data = json.loads(result_raw)

    assert data["status"] == "success"
    assert data["stock_by_location"] == {
        "Downtown": 0,
        "Uptown": 0,
        "Suburbs": 0,
    }
    assert "fallback_recommendations" in data["instruction"]


def test_check_local_inventory_demo_stockout_int_input():
    """Verify integer input is handled correctly for demo stockout ID."""
    result_raw = inventory.check_local_inventory(14325)
    data = json.loads(result_raw)

    assert data["status"] == "success"
    assert sum(data["stock_by_location"].values()) == 0


def test_check_local_inventory_normal_item():
    """Verify regular items return active inventory stock for all stores."""
    with patch("random.choice", side_effect=[5, 4, 8]):
        result_raw = inventory.check_local_inventory("10001")
        data = json.loads(result_raw)

        assert data["status"] == "success"
        assert data["stock_by_location"]["Downtown"] == 5
        assert data["stock_by_location"]["Uptown"] == 4
        assert data["stock_by_location"]["Suburbs"] == 8
        assert data["instruction"] == "Present the stock levels to the user."
