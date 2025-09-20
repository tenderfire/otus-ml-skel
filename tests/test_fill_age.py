import pytest
import pandas as pd
import numpy as np

from titanic.features.fill import age_imputer, fill_age, RandomAgeImputer


class TestRandomAgeImputer:
    """Tests for RandomAgeImputer class."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        imputer = RandomAgeImputer()
        assert imputer.min_age == 1
        assert imputer.max_age == 80
        assert imputer.random_state is None
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        imputer = RandomAgeImputer(min_age=10, max_age=60, random_state=42)
        assert imputer.min_age == 10
        assert imputer.max_age == 60
        assert imputer.random_state == 42
    
    def test_fit_returns_self(self):
        """Test that fit method returns self."""
        imputer = RandomAgeImputer()
        result = imputer.fit(pd.Series([1, 2, np.nan]))
        assert result is imputer
    
    def test_transform_no_missing_values(self):
        """Test transform when there are no missing values."""
        imputer = RandomAgeImputer(random_state=42)
        ages = pd.Series([25, 30, 45, 50])
        result = imputer.transform(ages)
        
        # Should return unchanged data
        pd.testing.assert_series_equal(result, ages)
    
    def test_transform_all_missing_values(self):
        """Test transform when all values are missing."""
        imputer = RandomAgeImputer(min_age=20, max_age=30, random_state=42)
        ages = pd.Series([np.nan, np.nan, np.nan])
        result = imputer.transform(ages)
        
        # Check that no NaN values remain
        assert not result.isna().any()
        
        # Check that all values are within expected range
        assert (result >= 20).all()
        assert (result <= 30).all()
        
        # Check that values are integers
        assert result.dtype in [np.int64, np.int32]
    
    def test_transform_mixed_values(self):
        """Test transform with mix of missing and non-missing values."""
        imputer = RandomAgeImputer(min_age=1, max_age=80, random_state=42)
        ages = pd.Series([25, np.nan, 45, np.nan, 50])
        result = imputer.transform(ages)
        
        # Check that no NaN values remain
        assert not result.isna().any()
        
        # Check that non-missing values are unchanged
        assert result.iloc[0] == 25
        assert result.iloc[2] == 45
        assert result.iloc[4] == 50
        
        # Check that filled values are within expected range
        assert 1 <= result.iloc[1] <= 80
        assert 1 <= result.iloc[3] <= 80
    
    def test_transform_reproducibility(self):
        """Test that transform produces reproducible results with same random_state."""
        ages = pd.Series([np.nan, np.nan, 25, np.nan])
        
        imputer1 = RandomAgeImputer(random_state=42)
        imputer2 = RandomAgeImputer(random_state=42)
        
        result1 = imputer1.transform(ages)
        result2 = imputer2.transform(ages)
        
        pd.testing.assert_series_equal(result1, result2)


class TestAgeImputer:
    """Tests for age_imputer function."""
    
    def test_age_imputer_default(self):
        """Test age_imputer function with default parameters."""
        imputer = age_imputer()
        assert isinstance(imputer, RandomAgeImputer)
        assert imputer.min_age == 1
        assert imputer.max_age == 80
        assert imputer.random_state is None
    
    def test_age_imputer_custom_params(self):
        """Test age_imputer function with custom parameters."""
        imputer = age_imputer(min_age=18, max_age=65, random_state=123)
        assert isinstance(imputer, RandomAgeImputer)
        assert imputer.min_age == 18
        assert imputer.max_age == 65
        assert imputer.random_state == 123


class TestFillAge:
    """Tests for fill_age function."""
    
    def test_fill_age_dataframe_input(self):
        """Test fill_age function with DataFrame input."""
        df = pd.DataFrame({
            'Age': [25, np.nan, 45, np.nan, 50],
            'Name': ['A', 'B', 'C', 'D', 'E']
        })
        
        result = fill_age(df, min_age=20, max_age=70, random_state=42)
        
        # Check that result is a Series
        assert isinstance(result, pd.Series)
        
        # Check that no NaN values remain
        assert not result.isna().any()
        
        # Check that non-missing values are unchanged
        assert result.iloc[0] == 25
        assert result.iloc[2] == 45
        assert result.iloc[4] == 50
        
        # Check that filled values are within expected range
        assert 20 <= result.iloc[1] <= 70
        assert 20 <= result.iloc[3] <= 70
    
    def test_fill_age_no_missing_values(self):
        """Test fill_age when DataFrame has no missing Age values."""
        df = pd.DataFrame({
            'Age': [25, 30, 45, 50],
            'Name': ['A', 'B', 'C', 'D']
        })
        
        result = fill_age(df, random_state=42)
        
        # Should return unchanged Age column
        pd.testing.assert_series_equal(result, df['Age'])
    
    def test_fill_age_all_missing_values(self):
        """Test fill_age when all Age values are missing."""
        df = pd.DataFrame({
            'Age': [np.nan, np.nan, np.nan],
            'Name': ['A', 'B', 'C']
        })
        
        result = fill_age(df, min_age=1, max_age=80, random_state=42)
        
        # Check that no NaN values remain
        assert not result.isna().any()
        
        # Check that all values are within expected range
        assert (result >= 1).all()
        assert (result <= 80).all()
    
    def test_fill_age_reproducibility(self):
        """Test that fill_age produces reproducible results."""
        df = pd.DataFrame({
            'Age': [25, np.nan, np.nan, 50],
            'Name': ['A', 'B', 'C', 'D']
        })
        
        result1 = fill_age(df, random_state=42)
        result2 = fill_age(df, random_state=42)
        
        pd.testing.assert_series_equal(result1, result2)
    
    @pytest.mark.parametrize("min_age,max_age", [(1, 80), (18, 65), (25, 40)])
    def test_fill_age_range_validation(self, min_age, max_age):
        """Test that filled ages are within specified range."""
        df = pd.DataFrame({
            'Age': [np.nan] * 100,  # Many missing values for better testing
            'Name': [f'Person_{i}' for i in range(100)]
        })
        
        result = fill_age(df, min_age=min_age, max_age=max_age, random_state=42)
        
        # Check that all values are within expected range
        assert (result >= min_age).all()
        assert (result <= max_age).all()
        
        # Check that we actually get some variety in the generated ages
        # (with 100 samples, we should get multiple different values)
        unique_values = result.nunique()
        assert unique_values > 1  # Should have more than 1 unique value