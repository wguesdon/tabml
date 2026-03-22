"""Advanced feature engineering utilities.

This module provides advanced feature engineering capabilities including
TF-IDF, date/time features, text statistics, and more sophisticated transformations.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import string
from loguru import logger
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import nltk
    from nltk.corpus import stopwords
    
    # Try to use stopwords, download if not present
    try:
        STOP_WORDS = set(stopwords.words('english'))
    except LookupError:
        # Download stopwords silently if not present
        logger.info("Downloading NLTK stopwords data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)  # Also download punkt tokenizer
        try:
            STOP_WORDS = set(stopwords.words('english'))
            logger.info("NLTK data downloaded successfully")
        except:
            logger.warning("Failed to download NLTK stopwords")
            STOP_WORDS = set()
except ImportError:
    logger.warning("NLTK not installed. Some text features will be limited.")
    STOP_WORDS = set()


class AdvancedFeatureEngineer:
    """Advanced feature engineering for complex transformations.
    
    Provides sophisticated feature engineering including TF-IDF, date features,
    text statistics, and interaction terms.
    
    Example:
        >>> engineer = AdvancedFeatureEngineer()
        >>> 
        >>> # Add date features
        >>> df = engineer.create_date_features(df, date_columns=['transaction_date'])
        >>> 
        >>> # Add TF-IDF features
        >>> df = engineer.create_tfidf_features(df, text_columns=['description'])
        >>> 
        >>> # Add text statistics
        >>> df = engineer.create_text_features(df, text_columns=['review'])
    """
    
    def __init__(self):
        """Initialize advanced feature engineer."""
        self.tfidf_vectorizers = {}
        self.svd_transformers = {}
        self.date_columns_processed = []
        self.text_columns_processed = []
        
    def create_date_features(self, df: pd.DataFrame, 
                            date_columns: List[str],
                            include_cyclical: bool = True,
                            include_lag: bool = True,
                            drop_original: bool = True) -> pd.DataFrame:
        """Create comprehensive date/time features.
        
        Args:
            df: Input DataFrame
            date_columns: List of date column names
            include_cyclical: Whether to include sin/cos transformations
            include_lag: Whether to include lag features
            drop_original: Whether to drop original date columns
            
        Returns:
            DataFrame with date features added
            
        Example:
            >>> df = engineer.create_date_features(
            ...     df, 
            ...     date_columns=['order_date', 'ship_date'],
            ...     include_cyclical=True
            ... )
        """
        df = df.copy()
        
        for col in date_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
                
            # Convert to datetime
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Basic date components
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_dayofyear'] = df[col].dt.dayofyear
            df[f'{col}_weekofyear'] = df[col].dt.isocalendar().week
            df[f'{col}_quarter'] = df[col].dt.quarter
            df[f'{col}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)
            df[f'{col}_is_month_start'] = df[col].dt.is_month_start.astype(int)
            df[f'{col}_is_month_end'] = df[col].dt.is_month_end.astype(int)
            df[f'{col}_is_quarter_start'] = df[col].dt.is_quarter_start.astype(int)
            df[f'{col}_is_quarter_end'] = df[col].dt.is_quarter_end.astype(int)
            df[f'{col}_is_year_start'] = df[col].dt.is_year_start.astype(int)
            df[f'{col}_is_year_end'] = df[col].dt.is_year_end.astype(int)
            
            # Time components if datetime has time
            if df[col].dt.time.ne(pd.Timestamp('00:00:00').time()).any():
                df[f'{col}_hour'] = df[col].dt.hour
                df[f'{col}_minute'] = df[col].dt.minute
                df[f'{col}_second'] = df[col].dt.second
                df[f'{col}_is_morning'] = ((df[col].dt.hour >= 6) & (df[col].dt.hour < 12)).astype(int)
                df[f'{col}_is_afternoon'] = ((df[col].dt.hour >= 12) & (df[col].dt.hour < 18)).astype(int)
                df[f'{col}_is_evening'] = ((df[col].dt.hour >= 18) & (df[col].dt.hour < 24)).astype(int)
                df[f'{col}_is_night'] = ((df[col].dt.hour >= 0) & (df[col].dt.hour < 6)).astype(int)
            
            # Cyclical features
            if include_cyclical:
                df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[f'{col}_month'] / 12)
                df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[f'{col}_month'] / 12)
                df[f'{col}_day_sin'] = np.sin(2 * np.pi * df[f'{col}_day'] / 31)
                df[f'{col}_day_cos'] = np.cos(2 * np.pi * df[f'{col}_day'] / 31)
                df[f'{col}_dayofweek_sin'] = np.sin(2 * np.pi * df[f'{col}_dayofweek'] / 7)
                df[f'{col}_dayofweek_cos'] = np.cos(2 * np.pi * df[f'{col}_dayofweek'] / 7)
                df[f'{col}_quarter_sin'] = np.sin(2 * np.pi * df[f'{col}_quarter'] / 4)
                df[f'{col}_quarter_cos'] = np.cos(2 * np.pi * df[f'{col}_quarter'] / 4)
            
            # Lag features
            if include_lag:
                # Days since epoch
                df[f'{col}_days_since_epoch'] = (df[col] - pd.Timestamp('1970-01-01')).dt.days
                
                # Differences between consecutive dates
                df[f'{col}_diff_days'] = df[col].diff().dt.days
                df[f'{col}_diff_hours'] = df[col].diff().dt.total_seconds() / 3600
            
            # Drop original column if requested
            if drop_original:
                df = df.drop(columns=[col])
                
            self.date_columns_processed.append(col)
            
        logger.info(f"Created date features for {len(date_columns)} columns")
        return df
    
    def create_tfidf_features(self, df: pd.DataFrame,
                              text_columns: List[str],
                              max_features: int = 100,
                              n_components: int = 10,
                              analyzer: str = 'word',
                              ngram_range: Tuple[int, int] = (1, 1),
                              min_df: Union[int, float] = 1,
                              max_df: Union[int, float] = 1.0) -> pd.DataFrame:
        """Create TF-IDF features with SVD dimensionality reduction.
        
        Args:
            df: Input DataFrame
            text_columns: List of text column names
            max_features: Maximum number of TF-IDF features
            n_components: Number of SVD components
            analyzer: 'word', 'char', or 'char_wb'
            ngram_range: Tuple of (min_n, max_n) for n-grams
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            
        Returns:
            DataFrame with TF-IDF features added
            
        Example:
            >>> df = engineer.create_tfidf_features(
            ...     df,
            ...     text_columns=['description', 'title'],
            ...     max_features=500,
            ...     n_components=20,
            ...     ngram_range=(1, 2)
            ... )
        """
        df = df.copy()
        
        for col in tqdm(text_columns, desc="Creating TF-IDF features"):
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            # Fill missing values
            df[col] = df[col].fillna("")
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                analyzer=analyzer,
                ngram_range=ngram_range,
                min_df=min_df,
                max_df=max_df,
                stop_words='english' if analyzer == 'word' else None
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(df[col])
            
            # Apply SVD for dimensionality reduction
            svd = TruncatedSVD(n_components=min(n_components, tfidf_matrix.shape[1]))
            tfidf_svd = svd.fit_transform(tfidf_matrix)
            
            # Create column names
            feature_names = [f"{col}_tfidf_{i}" for i in range(tfidf_svd.shape[1])]
            
            # Add to dataframe
            tfidf_df = pd.DataFrame(tfidf_svd, columns=feature_names, index=df.index)
            df = pd.concat([df, tfidf_df], axis=1)
            
            # Store transformers for later use
            self.tfidf_vectorizers[col] = vectorizer
            self.svd_transformers[col] = svd
            
            # Optionally drop original column
            # df = df.drop(columns=[col])
            
        logger.info(f"Created TF-IDF features for {len(text_columns)} columns")
        return df
    
    def create_text_features(self, df: pd.DataFrame,
                            text_columns: List[str]) -> pd.DataFrame:
        """Create statistical features from text columns.
        
        Args:
            df: Input DataFrame
            text_columns: List of text column names
            
        Returns:
            DataFrame with text statistics added
            
        Example:
            >>> df = engineer.create_text_features(
            ...     df,
            ...     text_columns=['review', 'comment']
            ... )
        """
        df = df.copy()
        
        for col in tqdm(text_columns, desc="Creating text features"):
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            # Fill missing values
            df[col] = df[col].fillna("")
            
            # Basic statistics
            df[f'{col}_length'] = df[col].str.len()
            df[f'{col}_word_count'] = df[col].str.split().str.len()
            df[f'{col}_char_count'] = df[col].apply(lambda x: sum(len(word) for word in str(x).split()))
            
            # Average word length
            df[f'{col}_avg_word_length'] = df[f'{col}_char_count'] / df[f'{col}_word_count'].replace(0, 1)
            
            # Punctuation and special characters
            df[f'{col}_punctuation_count'] = df[col].apply(
                lambda x: sum(1 for char in str(x) if char in string.punctuation)
            )
            df[f'{col}_digit_count'] = df[col].apply(
                lambda x: sum(1 for char in str(x) if char.isdigit())
            )
            df[f'{col}_upper_count'] = df[col].apply(
                lambda x: sum(1 for char in str(x) if char.isupper())
            )
            df[f'{col}_special_char_count'] = df[col].apply(
                lambda x: sum(1 for char in str(x) if not char.isalnum() and not char.isspace())
            )
            
            # Stop words (if NLTK available)
            if STOP_WORDS:
                df[f'{col}_stopword_count'] = df[col].apply(
                    lambda x: sum(1 for word in str(x).lower().split() if word in STOP_WORDS)
                )
                df[f'{col}_non_stopword_count'] = df[f'{col}_word_count'] - df[f'{col}_stopword_count']
                df[f'{col}_stopword_ratio'] = df[f'{col}_stopword_count'] / df[f'{col}_word_count'].replace(0, 1)
            
            # Unique words
            df[f'{col}_unique_word_count'] = df[col].apply(lambda x: len(set(str(x).split())))
            df[f'{col}_lexical_diversity'] = df[f'{col}_unique_word_count'] / df[f'{col}_word_count'].replace(0, 1)
            
            # Sentence count (approximate)
            df[f'{col}_sentence_count'] = df[col].str.count(r'[.!?]+')
            df[f'{col}_avg_sentence_length'] = df[f'{col}_word_count'] / df[f'{col}_sentence_count'].replace(0, 1)
            
            # Question and exclamation marks
            df[f'{col}_question_count'] = df[col].str.count(r'\?')
            df[f'{col}_exclamation_count'] = df[col].str.count(r'!')
            
            # Capitalized words
            df[f'{col}_capitalized_count'] = df[col].apply(
                lambda x: sum(1 for word in str(x).split() if word and word[0].isupper())
            )
            
            # All caps words
            df[f'{col}_all_caps_count'] = df[col].apply(
                lambda x: sum(1 for word in str(x).split() if word.isupper() and len(word) > 1)
            )
            
            self.text_columns_processed.append(col)
            
        logger.info(f"Created text features for {len(text_columns)} columns")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   numeric_columns: Optional[List[str]] = None,
                                   max_interactions: int = 50) -> pd.DataFrame:
        """Create interaction features between numeric columns.
        
        Args:
            df: Input DataFrame
            numeric_columns: List of numeric columns (if None, auto-detect)
            max_interactions: Maximum number of interactions to create
            
        Returns:
            DataFrame with interaction features added
        """
        df = df.copy()
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit columns if too many
        if len(numeric_columns) > 10:
            # Select most important based on variance
            variances = df[numeric_columns].var()
            numeric_columns = variances.nlargest(10).index.tolist()
        
        interactions_created = 0
        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                if interactions_created >= max_interactions:
                    break
                    
                col1, col2 = numeric_columns[i], numeric_columns[j]
                
                # Multiplication
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Division (with protection against division by zero)
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2].replace(0, 1))
                
                # Addition
                df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                
                # Subtraction
                df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                
                interactions_created += 4
                
        logger.info(f"Created {interactions_created} interaction features")
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame,
                                  numeric_columns: Optional[List[str]] = None,
                                  degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for numeric columns.
        
        Args:
            df: Input DataFrame
            numeric_columns: List of numeric columns (if None, auto-detect)
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features added
        """
        df = df.copy()
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit columns if too many
        if len(numeric_columns) > 10:
            # Select most important based on variance
            variances = df[numeric_columns].var()
            numeric_columns = variances.nlargest(10).index.tolist()
        
        for col in numeric_columns:
            for d in range(2, degree + 1):
                df[f'{col}_pow_{d}'] = df[col] ** d
                
            # Also add sqrt and log transformations
            df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
            df[f'{col}_log'] = np.log1p(np.abs(df[col]))
            
        logger.info(f"Created polynomial features for {len(numeric_columns)} columns")
        return df
    
    def create_aggregate_features(self, df: pd.DataFrame,
                                 group_columns: List[str],
                                 agg_columns: List[str],
                                 agg_functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """Create aggregate features based on grouping.
        
        Args:
            df: Input DataFrame
            group_columns: Columns to group by
            agg_columns: Columns to aggregate
            agg_functions: Aggregation functions to apply
            
        Returns:
            DataFrame with aggregate features added
        """
        df = df.copy()
        
        for group_col in group_columns:
            for agg_col in agg_columns:
                for func in agg_functions:
                    feature_name = f'{agg_col}_by_{group_col}_{func}'
                    
                    # Calculate aggregation
                    agg_values = df.groupby(group_col)[agg_col].transform(func)
                    df[feature_name] = agg_values
                    
                    # Also create diff from aggregation
                    df[f'{feature_name}_diff'] = df[agg_col] - agg_values
                    
        logger.info(f"Created aggregate features for {len(group_columns)} groups")
        return df
    
    def create_multi_column_tfidf(self, train: pd.DataFrame, test: pd.DataFrame,
                                  text_columns: List[str],
                                  max_features: int = 3000,
                                  analyzer: str = 'char_wb') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create TF-IDF features for multiple columns (train/test aware).
        
        Args:
            train: Training DataFrame
            test: Test DataFrame
            text_columns: List of text columns
            max_features: Maximum features per column
            analyzer: Type of analyzer ('word', 'char', 'char_wb')
            
        Returns:
            Tuple of (train_with_features, test_with_features)
        """
        train_features = []
        test_features = []
        
        for col in tqdm(text_columns, desc="Processing text columns"):
            # Fill missing values
            train[col] = train[col].fillna("")
            test[col] = test[col].fillna("")
            
            # Create and fit vectorizer
            vectorizer = TfidfVectorizer(
                analyzer=analyzer,
                max_features=max_features
            )
            
            # Fit on train, transform both
            train_tfidf = vectorizer.fit_transform(train[col])
            test_tfidf = vectorizer.transform(test[col])
            
            # Convert to DataFrames
            feature_names = [f"tfidf_{col}_{i}" for i in range(train_tfidf.shape[1])]
            train_tfidf_df = pd.DataFrame(
                train_tfidf.toarray(),
                columns=feature_names,
                index=train.index
            )
            test_tfidf_df = pd.DataFrame(
                test_tfidf.toarray(),
                columns=feature_names,
                index=test.index
            )
            
            train_features.append(train_tfidf_df)
            test_features.append(test_tfidf_df)
        
        # Concatenate all features
        train_with_features = pd.concat([train, *train_features], axis=1)
        test_with_features = pd.concat([test, *test_features], axis=1)
        
        return train_with_features, test_with_features