from plotnine import facet_wrap
import pandas as pd
from plotnine import ggplot, aes, geom_line, geom_vline, labs


def save_plots(df_cv, output_paths, ts, models: list[str], n_obs: int = 30):
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

    value_vars = ['y'] + [col for col in models if col in df_cv.columns]

    # For each ticker (unique_id), plot the full series and all forecasts
    run_dir = output_paths["run_dir"]
    tickers = list(df_cv['unique_id'].unique())
    # Add synthetic 'TOTAL' ticker for sum of all tickers
    if 'TOTAL' not in tickers:
        tickers.append('TOTAL')

    for ticker in tickers:
        if ticker == 'TOTAL':
            # Sum across all tickers for each ds
            ts_total = ts.groupby('ds', as_index=False)['y'].sum().sort_values('ds')
            ts_total['unique_id'] = 'TOTAL'
            ts_total = ts_total[['ds', 'unique_id', 'y']]
            ts_ticker = ts_total
        else:
            ts_ticker = ts[ts['unique_id'] == ticker][['ds', 'y']].drop_duplicates().sort_values('ds')

        ts_ticker = ts_ticker.iloc[-n_obs:] if n_obs > 0 else ts_ticker
        df_actuals = ts_ticker.copy()
        df_actuals['model'] = 'y'
        df_actuals = df_actuals[['ds', 'model', 'y']].rename(columns={'y': 'value'})

        # Forecasts: for each model, concatenate all cutoffs
        if ticker == 'TOTAL':
            df_ticker = df_cv[df_cv['unique_id'] == 'TOTAL'].copy()
        else:
            df_ticker = df_cv[df_cv['unique_id'] == ticker].copy()
        forecast_vars = [col for col in value_vars if col != 'y']
        df_forecasts = df_ticker.melt(
            id_vars=['ds', 'cutoff'],
            value_vars=forecast_vars,
            var_name='model',
            value_name='value'
        )
        # Remove duplicates (if any) for the same ds/model (keep latest cutoff)
        df_forecasts = df_forecasts.sort_values('ds').drop_duplicates(['ds', 'model', 'cutoff'], keep='last')

        cutoffs = df_ticker['cutoff'].dropna().unique()
        if len(cutoffs) > 0:
            df_actuals = (
                df_actuals.assign(_key=1)
                .merge(pd.DataFrame({'cutoff': cutoffs, '_key': 1}), on='_key')
                .drop(columns=['_key'])
            )
        else:
            df_actuals['cutoff'] = pd.NaT

        # Combine actuals and forecasts
        df_long = pd.concat([df_actuals, df_forecasts], ignore_index=True)

        if not df_forecasts.empty:
            cutoff_bounds = (
                df_forecasts.groupby('cutoff', as_index=False)['ds']
                .agg(['min', 'max'])
                .reset_index()
            )
            cutoff_lines = pd.concat(
                [
                    cutoff_bounds[['cutoff', 'min']].rename(columns={'min': 'xintercept'}),
                    cutoff_bounds[['cutoff', 'max']].rename(columns={'max': 'xintercept'}),
                ],
                ignore_index=True,
            )
        else:
            cutoff_lines = pd.DataFrame({'cutoff': [], 'xintercept': []})

        # Plot: Forecast vs Actuals for this ticker (facet per cutoff)
        p = (
            ggplot(df_long)
            + geom_line(aes(x='ds', y='value', color='model'))
            + geom_vline(
                aes(xintercept='xintercept'),
                data=cutoff_lines,
                linetype='dotted',
                color='black',
                size=0.4,
            )
            + labs(title=f'Forecast vs Actuals: {ticker}', x='Date', y='Value')
            + facet_wrap('~cutoff')
        )
        out_path = run_dir / f'forecast_vs_actuals_{ticker}.png'
        p.save(str(out_path))
