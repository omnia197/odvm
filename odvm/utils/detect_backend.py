def detect_backend(df):
    import pandas as pd
    try:
        import dask.dataframe as dd
    except ImportError:
        dd = None

    if dd and isinstance(df, dd.DataFrame):
        return "dask"
    elif isinstance(df, pd.DataFrame):
        return "pandas"
    else:
        return "unknown"
