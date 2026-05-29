
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

def save_plots(df_cv, output_paths, config_path="config.yaml"):
    """
    Save forecast vs actuals plots as PNG files using plotnine.
    Args:
        df_cv: DataFrame with columns ['unique_id', 'ds', 'cutoff', 'y', 'AutoNHITS', 'AutoNBEATS', ...]
        output_paths: dict with keys for output file paths (e.g., 'forecast_vs_actuals', 'scatter_forecast_vs_actuals')
    """
    # Ensure datetime columns
    df_cv = df_cv.copy()
    df_cv['ds'] = pd.to_datetime(df_cv['ds'])
    df_cv['cutoff'] = pd.to_datetime(df_cv['cutoff'])

    # Get models from config.yaml
    models = get_models_from_config(config_path)
    value_vars = ['y'] + [col for col in models if col in df_cv.columns]

    # For each ticker (unique_id), plot the full series and all forecasts
    run_dir = output_paths["run_dir"]
    for ticker in df_cv['unique_id'].unique():
        df_ticker = df_cv[df_cv['unique_id'] == ticker].copy()
        df_long = df_ticker.melt(
            id_vars=['ds', 'cutoff'],
            value_vars=value_vars,
            var_name='model',
            value_name='value'
        )

        # Plot: Forecast vs Actuals for this ticker
        p = (
            ggplot(df_long)
            + geom_line(aes(x='ds', y='value', color='model'))
            + facet_wrap('~cutoff', scales='free_y')
            + labs(title=f'Forecast vs Actuals: {ticker}', x='Date', y='Value')
        )
        out_path = run_dir / f'forecast_vs_actuals_{ticker}.png'
        p.save(str(out_path))
