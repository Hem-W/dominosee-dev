"""
Test fixtures for dominosee tests
"""
import os
from pathlib import Path

import pytest
import xarray as xr


@pytest.fixture
def ds_example():
    """
    Load the example dataset used for testing event extraction
    
    Returns
    -------
    xarray.Dataset
        Example dataset with Tmean and other variables
    """
    # Find the data file path - try both the tests/data dir and project root
    data_path = Path(__file__).parent / "data" / "test_CC_Example_Data1.nc"
    if not data_path.exists():
        # Try project root as fallback
        root_path = Path(__file__).parent.parent / "test_CC_Example_Data1.nc"
        if root_path.exists():
            data_path = root_path
        else:
            raise FileNotFoundError(
                "Could not find test_CC_Example_Data1.nc in tests/data/ or project root"
            )
    
    # Load the dataset
    return xr.open_dataset(data_path)
