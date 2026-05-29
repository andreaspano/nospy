
import yaml
import pandas as pd
from plotnine import ggplot, aes, geom_line, facet_wrap, facet_grid, labs, geom_point, geom_abline

def get_models_from_config(config_path="config.yaml"):
    """
    Load the list of models from config.yaml.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("models", [])

def save_plots(df_cv, output_paths, ts, config_path="config.yaml", n_obs: int = 30):
    """
    Save forecast vs actuals plots as PNG files using plotnine.
    Args:
        df_cv: DataFrame with columns ['unique_id', 'ds', 'cutoff', 'y', 'AutoNHITS', 'AutoNBEATS', ...]
        output_paths: dict with keys for output file paths (e.g., 'forecast_vs_actuals', 'scatter_forecast_vs_actuals')
        ts: DataFrame with columns ['unique_id', 'ds', 'y'] (full time series for all tickers)
        n_obs: Number of most recent historical observations to show (default 30)
    """
    # Ensure datetime columns
    df_cv = df_cv.copy()
    df_cv['ds'] = pd.to_datetime(df_cv['ds'])
    df_cv['cutoff'] = pd.to_datetime(df_cv['cutoff'])

    ts = ts.copy()
    ts['ds'] = pd.to_datetime(ts['ds'])

    # Get models from config.yaml
    models = get_models_from_config(config_path)
    value_vars = ['y'] + [col for col in models if col in df_cv.columns]

    # For each ticker (unique_id), plot the full series and all forecasts
    run_dir = output_paths["run_dir"]
    tickers = list(df_cv['unique_id'].unique())
    # Add synthetic 'mib' ticker for sum of all tickers
    if 'mib' not in tickers:
        tickers.append('mib')

    for ticker in tickers:
        if ticker == 'mib':
            # Sum across all tickers for each ds
            ts_mib = ts.groupby('ds', as_index=False)['y'].sum().sort_values('ds')
            ts_mib['unique_id'] = 'mib'
            ts_mib = ts_mib[['ds', 'unique_id', 'y']]
            ts_ticker = ts_mib
        else:
            ts_ticker = ts[ts['unique_id'] == ticker][['ds', 'y']].drop_duplicates().sort_values('ds')

        ts_ticker = ts_ticker.iloc[-n_obs:] if n_obs > 0 else ts_ticker
        df_actuals = ts_ticker.copy()
        df_actuals['model'] = 'y'
        df_actuals = df_actuals[['ds', 'model', 'y']].rename(columns={'y': 'value'})

        # Forecasts: for each model, concatenate all cutoffs
        if ticker == 'mib':
            df_ticker = df_cv[df_cv['unique_id'] == 'mib'].copy()
        else:
            df_ticker = df_cv[df_cv['unique_id'] == ticker].copy()
        forecast_vars = [col for col in value_vars if col != 'y']
        df_forecasts = df_ticker.melt(
            id_vars=['ds'],
            value_vars=forecast_vars,
            var_name='model',
            value_name='value'
        )
        # Remove duplicates (if any) for the same ds/model (keep latest cutoff)
        df_forecasts = df_forecasts.sort_values('ds').drop_duplicates(['ds', 'model'], keep='last')

        # Combine actuals and forecasts
        df_long = pd.concat([df_actuals, df_forecasts], ignore_index=True)

        # Plot: Forecast vs Actuals for this ticker (no facet)
        p = (
            ggplot(df_long)
            + geom_line(aes(x='ds', y='value', color='model'))
            + labs(title=f'Forecast vs Actuals: {ticker}', x='Date', y='Value')
        )
        out_path = run_dir / f'forecast_vs_actuals_{ticker}.png'
        p.save(str(out_path))
