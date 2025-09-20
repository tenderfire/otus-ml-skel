import pandas
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


__all__ = ["embarked_imputer", "fill_embarked", "age_imputer", "fill_age"]


def embarked_imputer() -> SimpleImputer:
    return SimpleImputer(strategy="most_frequent")


def fill_embarked(df: pandas.DataFrame) -> pandas.Series:
    return embarked_imputer().fit_transform(df[["Embarked"]])


class RandomAgeImputer(BaseEstimator, TransformerMixin):
    """Custom imputer that fills missing Age values with random integers between min_age and max_age."""
    
    def __init__(self, min_age: int = 1, max_age: int = 80, random_state: int = None):
        self.min_age = min_age
        self.max_age = max_age
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """Fit method - no fitting required for random imputation."""
        return self
    
    def transform(self, X):
        """Transform method - fill missing values with random ages."""
        X_copy = X.copy()
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Find missing values
        missing_mask = X_copy.isna()
        n_missing = missing_mask.sum()
        
        if n_missing > 0:
            # Generate random ages for missing values
            random_ages = np.random.randint(self.min_age, self.max_age + 1, size=n_missing)
            X_copy[missing_mask] = random_ages
            # Convert to integer type if all values are filled
            if not X_copy.isna().any():
                X_copy = X_copy.astype(int)
        
        return X_copy


def age_imputer(min_age: int = 1, max_age: int = 80, random_state: int = None) -> RandomAgeImputer:
    """Create a RandomAgeImputer instance for filling missing Age values.
    
    Args:
        min_age: Minimum age for random generation (default: 1)
        max_age: Maximum age for random generation (default: 80)
        random_state: Random seed for reproducibility (default: None)
    
    Returns:
        RandomAgeImputer instance
    """
    return RandomAgeImputer(min_age=min_age, max_age=max_age, random_state=random_state)


def fill_age(df: pandas.DataFrame, min_age: int = 1, max_age: int = 80, random_state: int = None) -> pandas.Series:
    """Fill missing Age values with random integers between min_age and max_age.
    
    Args:
        df: DataFrame containing Age column
        min_age: Minimum age for random generation (default: 1)
        max_age: Maximum age for random generation (default: 80)
        random_state: Random seed for reproducibility (default: None)
    
    Returns:
        Series with filled Age values
    """
    imputer = age_imputer(min_age=min_age, max_age=max_age, random_state=random_state)
    return imputer.fit_transform(df["Age"])
