import pandas as pd

file_cv = 'out/2026_05_29_09_36_cv.csv'

df = pd.read_csv(file_cv)


#convert ds and cutoff to datetime
df['ds'] = pd.to_datetime(df['ds'])
df['cutoff'] = pd.to_datetime(df['cutoff'])

from plotnine import ggplot, aes, geom_line, facet_wrap, labs, geom_point, geom_abline

# make long format for plotting
df_long = df.melt(
    id_vars=['unique_id', 'ds', 'cutoff'],
    value_vars=['y', 'AutoNHITS', 'AutoNBEATS'],
    var_name='model',
    value_name='value'
)

pl = (ggplot(df_long) 
    + geom_line(aes(x='ds', y='value', color='model')) 
    + facet_wrap('~unique_id', scales='free_y')
    + labs(title='Forecast vs Actuals', x='Date', y='Value')
    )

pl.show()


pl = (ggplot(df) 
    + geom_point(aes(x='y', y='AutoNHITS'), color='red', size=1.5) 
    + geom_point(aes(x='y', y='AutoNBEATS'), color='blue', size=1.5)
    + facet_wrap('~unique_id', scales='free')
    + labs(title='Forecast vs Actuals', x='Date', y='Value')
    + geom_abline(slope=1, intercept=0, linetype='dashed', color='gray')
)

pl.show()
