import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from statsmodels.tsa.stattools import pacf, acf
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error


def create_lag_features(y, name, scaler=True, thres=0.2):
    '''
    Lag features: they use the original time series itself as feature with a certain shift usually called lag.
    Lags can be chosen automatically looking at the values of the partial autocorrelation function.
    In particular, we take as features only the lags where the PACF is greater than 0.2, equivalent to a 5% 
    relevance for the lag
    '''
    
    features = pd.DataFrame()
    
    # partial = pd.Series(data=pacf(y, nlags=48))
    partial = pd.Series(data=pacf(y))
    lags = list(partial[np.abs(partial) >= thres].index)
    
    df = pd.DataFrame()
    
    # avoid to insert the time series itself
    # lags.remove(0)
    
    for l in lags:
        df[f"{name}_lag_{l}"] = y.shift(l)
    
    if scaler:
        scaler = StandardScaler()
        features = pd.DataFrame(scaler.fit_transform(df[df.columns]),
                                columns=df.columns)
    else:
        features = pd.DataFrame(df[df.columns],
                                columns=df.columns)
    features.index = y.index
    
    return features

def plot_corr_matrix(dataframe):
    corr=np.abs(dataframe.corr(numeric_only=True))

    # Set up mask for triangle representation
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(14, 14))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask,  vmax=1,square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=corr)

    plt.show()

def print_scores(model, X_train, X_test, y_train, y_test):
    print(f'Test score: {model.score(X_test, y_test)*100:.2f}%')
    print(f'Train score: {model.score(X_train, y_train)*100:.2f}%')
    print(f'Overfitting test: {(model.score(X_train, y_train) - model.score(X_test, y_test))*100:.2f}%')
    y_pred = model.predict(X_test)
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}')

def features_importance(model):
    feature_importance = pd.DataFrame(data=model.feature_importances_,
                                        index=model.feature_names_in_,
                                        columns=['importance']).sort_values('importance', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=feature_importance['importance'],
            y=feature_importance.index,
            orientation='h'
        )
    )

    fig.update_layout(title='Feature importance', width=800, height=600)
    fig.show()

    # feature_importance.plot(kind='barh', title='Features importance')
    # plt.show()

def plot_true_pred(y_train, y_test, y_fore, title='Title', forecast=False):

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Scatter(
            x=y_train.index,
            y=y_train,
            name='Observed' if forecast else 'Train'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_test.index,
            y=y_test,
            name='Unseen' if forecast else 'Test'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_fore.index,
            y=y_fore,
            name='Predict'
        )
    )
    fig.update_layout(title=title)
    fig.show()


def create_corr_plot(series, plot_pacf=False):
    corr_array = pacf(series.dropna(), alpha=0.05) if plot_pacf else acf(series.dropna(), alpha=0.05)
    lower_y = corr_array[1][:,0] - corr_array[0]
    upper_y = corr_array[1][:,1] - corr_array[0]

    fig = go.Figure()
    [fig.add_scatter(x=(x,x), y=(0,corr_array[0][x]), 
                     mode='lines',
                     line_color='#3f3f3f'
                     ) 
                     for x in range(len(corr_array[0]))
                     ]
    
    fig.add_scatter(x=np.arange(len(corr_array[0])), 
                    y=corr_array[0], 
                    mode='markers', 
                    marker_color='#1f77b4',
                    marker_size=12)
    
    fig.add_scatter(x=np.arange(len(corr_array[0])),
                    y=upper_y, mode='lines', 
                    line_color='rgba(255,255,255,0)')
    
    fig.add_scatter(x=np.arange(len(corr_array[0])), 
                    y=lower_y, 
                    mode='lines',
                    fillcolor='rgba(32, 146, 230,0.3)',
                    fill='tonexty', 
                    line_color='rgba(255,255,255,0)')
    
    fig.update_traces(showlegend=False)

    fig.update_xaxes(range=[-1,len(corr_array[0])])

    fig.update_yaxes(zerolinecolor='#000000')
    
    title='Partial Autocorrelation (PACF)' if plot_pacf else 'Autocorrelation (ACF)'

    fig.update_layout(title=title)
    fig.update_xaxes(title_text='Lag')
    fig.update_yaxes(title_text='PACF value' if plot_pacf else 'ACF value')

    fig.show()



def create_decomposition_plots(series, model='additive'):
    trend = seasonal_decompose(x=series, model=model).trend
    seasonal = seasonal_decompose(x=series, model=model).seasonal
    residual = seasonal_decompose(x=series, model=model).resid

    tmp = [trend, seasonal, residual]
    decompose_df = pd.DataFrame(tmp).T
    decompose_df.columns = ['trend', 'seasonal', 'resid']

    fig = make_subplots(rows=4, cols=1)
    
    fig.append_trace(
        go.Scatter(
            x=decompose_df.index,
            y=series,
            name='Observed'
            ),
            row=1,
            col=1
    )

    fig.append_trace(
        go.Scatter(
            x=decompose_df.index,
            y=decompose_df.trend,
            name='Trend'
            ),
            row=2,
            col=1
    )

    fig.append_trace(
        go.Scatter(
            x=decompose_df.index,
            y=decompose_df.seasonal,
            name='Seasonal'
            ),
            row=3,
            col=1
    )

    fig.append_trace(
        go.Scatter(
            x=decompose_df.index,
            y=decompose_df.resid,
            mode='markers',
            name='Residual'
            ),
            row=4,
            col=1
    )

    fig.update_layout(height=800, width=800, title_text="Decomposition Plot")
    return fig