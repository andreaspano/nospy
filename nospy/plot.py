
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
    # Melt once for both plots
    df_long = df_cv.melt(
        id_vars=['unique_id', 'ds', 'cutoff'],
        value_vars=value_vars,
        var_name='model',
        value_name='value'
    )

    # Line plot: Forecast vs Actuals
    pl1 = (
        ggplot(df_long)
        + geom_line(aes(x='ds', y='value', color='model'))
        + facet_grid('cutoff~unique_id', scales='free_y')
        + labs(title='Forecast vs Actuals', x='Date', y='Value')
    )
    pl1.save(output_paths.get('forecast_vs_actuals', 'forecast_vs_actuals.png'))

    # Scatter plot: Forecast vs Actuals for all models (excluding 'y' as forecast)
    df_scatter = df_long[df_long['model'] != 'y'].copy()
    if not df_scatter.empty:
        # Merge to get actuals for each row
        df_scatter = df_scatter.merge(
            df_long[df_long['model'] == 'y'][['unique_id', 'ds', 'cutoff', 'value']],
            on=['unique_id', 'ds', 'cutoff'],
            suffixes=('', '_actual')
        )
        pl2 = (
            ggplot(df_scatter)
            + geom_point(aes(x='value_actual', y='value', color='model'), size=1.5)
            + facet_wrap('~unique_id', scales='free')
            + labs(title='Forecast vs Actuals', x='Actual', y='Forecast')
            + geom_abline(slope=1, intercept=0, linetype='dashed', color='gray')
        )
        pl2.save(output_paths.get('scatter_forecast_vs_actuals', 'scatter_forecast_vs_actuals.png'))
