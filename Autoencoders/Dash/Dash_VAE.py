# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
import mdl_utilities as mdl
from dash.exceptions import PreventUpdate

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

data_train, data_test, colors_test, colors_train = mdl.load_data()
model = mdl.load_model()
df_umap = mdl.umap_plot(data_test,colors_test)

fig1 = px.scatter(df_umap,x='Dim 1', y='Dim 2', color="Subgroups", title="Original Data")
fig2 = px.scatter(df_umap,x='Dim 1', y='Dim 2', color="Subgroups", title="Original Data")

app.layout = html.Div(children=[
    html.H1(children='Medulloblatoma visualization'),
    html.Div(children='''
        UMAP is a visualization tool, not analytic tool. It's a stochastic algorithm, so results may vary.
    '''),
    html.Div(["Work with Cavalli or Northcott data", 
              dcc.Checklist(id='dataset',
                    options=[
                        {'label': 'Cavalli', 'value': 'Cavalli'},
                        {'label': 'Northcott', 'value': 'Northcott'},
                    ],
                    value=['Northcott']
            )]),
    html.Div(["Generated data similar to which point: ",
              dcc.Input(id='my-input', value='0', type='number'),
              "How many points: ",
              dcc.Input(id='number_data', value='0', type='number')]),
    html.Div(["Generate data? ", 
                  daq.BooleanSwitch(
        id='generate',
        on=False),
    html.Div(id='boolean-switch-input0')
    ]),

    dcc.Graph(
        id='generated-data',
        figure=fig1
    ),
    
    html.Div(["Interpolate data between points: ",
              dcc.Input(id='interpolation-data1', value='0', type='number'),
              dcc.Input(id='interpolation-data2', value='0', type='number'),
             "How many steps",
              dcc.Input(id='interpolation-steps', value='0', type='number')]),
    html.Div(["Perform the interpolation? ", 
                  daq.BooleanSwitch(
        id='interpolate',
        on=False),
    html.Div(id='boolean-switch-input')
    ]),
    
    dcc.Graph(
        id='interpolated-data',
        figure=fig2
    )
    
])

@app.callback(
    Output('generated-data', 'figure'),
    Input('my-input', 'value'),
    Input('number_data', 'value'),
    Input('generate', 'on'),
    Input('dataset','value'))
def update_figure(selected_point,number,option,dataset):
    
    if 'Northcott' in dataset:
        data = data_test
        colors = colors_test
    elif 'Cavalli' in dataset:
        data = data_train
        colors = colors_train
    else:
        raise PreventUpdate
    
    if option == True:
        generated, colors_generated = mdl.data_generation(number,selected_point,data,colors,model)
        df_generated = mdl.umap_plot(generated,colors_generated)
    else:
        raise PreventUpdate

    fig = px.scatter(df_generated,x='Dim 1', y='Dim 2', color="Subgroups", title="Generated Data")

    fig.update_layout(transition_duration=500)

    return fig

@app.callback(
    Output('interpolated-data', 'figure'),
    Input('interpolation-data1', 'value'),
    Input('interpolation-data2', 'value'),
    Input('interpolation-steps', 'value'),
    Input('interpolate', 'on'),
    Input('dataset','value'))
def update_figure(origin,end,steps,option,dataset):
    
    if 'Northcott' in dataset:
        data = data_test
        colors = colors_test
    elif 'Cavalli' in dataset:
        data = data_train
        colors = colors_train
    else:
        raise PreventUpdate
    
    if option == True:
        generated, colors_generated = mdl.data_interpolation(steps,origin,end, colors, data, model)
        df_generated = mdl.umap_plot(generated,colors_generated)
    else:
        raise PreventUpdate

    fig = px.scatter(df_generated,x='Dim 1', y='Dim 2', color="Subgroups", title="Interpolated Data")

    fig.update_layout(transition_duration=500)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)