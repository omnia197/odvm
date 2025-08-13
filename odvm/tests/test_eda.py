import pandas as pd
import pytest
from ..eda.analyzer import EDAAnalyzer

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'age': [25, 30, 22, 40, None],
        'income': [50000, 60000, 55000, None, 52000],
        'gender': ['M', 'F', 'F', 'M', 'F'],
        'constant_col': [1, 1, 1, 1, 1]
    })

def test_get_shape(sample_df):
    analyzer = EDAAnalyzer(sample_df)
    assert analyzer.get_shape() == (5, 4)

def test_missing_values(sample_df):
    analyzer = EDAAnalyzer(sample_df)
    missing = analyzer.missing_values()
    assert missing['age'] == 1
    assert missing['income'] == 1

def test_constant_columns(sample_df):
    analyzer = EDAAnalyzer(sample_df)
    assert 'constant_col' in analyzer.constant_columns()

def test_columns_info(sample_df):
    analyzer = EDAAnalyzer(sample_df)
    info = analyzer.get_columns_info()
    assert info['age']['type'] == 'numerical'
    assert info['gender']['type'] == 'categorical'


from eda.visualizer import EDAVisualizer

def test_visualizer_runs_without_errors(sample_df, tmp_path):
    visualizer = EDAVisualizer(sample_df, save_dir=tmp_path)
    
    visualizer.plot_distributions()
    visualizer.plot_boxplots()
    visualizer.plot_correlation()
    visualizer.plot_target_distribution()
    visualizer.plot_pairplot()

    saved_files = list(tmp_path.glob("*.png"))
    assert len(saved_files) > 0
