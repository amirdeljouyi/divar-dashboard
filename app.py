import time
import importlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from inspect import signature


class Dataset:
    def __init__(self, df):
        self.df = df


def detect_outlier_zscore(data, threshold):
    outliers = pd.DataFrame([], columns=['ID', 'sqdist', 'cluster'])
    mean = np.mean(data.sqdist)
    std = np.std(data.sqdist)

    for y in data.itertuples():
        z_score = (y.sqdist - mean)/std
        if np.abs(z_score) > threshold:
            outliers = outliers.append(
                {'ID': y.ID, 'sqdist': y.sqdist, 'cluster': y.cluster}, ignore_index=True)
    return outliers


def detect_outlier_quantile(data, percent):
    quantile = data.sqdist.quantile(percent)
    outliers = pd.DataFrame([], columns=['ID', 'sqdist', 'cluster'])
    for y in data.itertuples():
        if y.sqdist > quantile:
            outliers = outliers.append(
                {'ID': y.ID, 'sqdist': y.sqdist, 'cluster': y.cluster}, ignore_index=True)
    return outliers


def outlierss(df, number_cluster, outlier_method, threshold):
    clusters = []
    outliers = pd.DataFrame([], columns=['ID', 'sqdist', 'cluster'])
    for i in range(number_cluster):
        clusters.append(df.loc[df.cluster == i])

        if outlier_method == "zsocre":
            outliers = outliers.append(detect_outlier_quantile(
                clusters[i], 0.75), ignore_index=True)
        else:
            outliers = outliers.append(detect_outlier_zscore(
                clusters[i], threshold), ignore_index=True)
    return outliers


def checkTrue(x, outliers, clusterNum):
    for i in range(clusterNum):
        if len(outliers.loc[outliers.ID == x]) != 0:
            return True
    return False


def dfAfterKmeans(kmeans, df):
    dist = kmeans.transform(df)**2
    df['sqdist'] = dist.sum(axis=1).round(2)
    df['cluster'] = kmeans.labels_
    df['ID'] = np.arange(len(df))
    return df


def NormalizeData(df, column_names):
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(df[column_names])
    scaled_df = pd.DataFrame(scaled_df, columns=column_names)
    return scaled_df


def generate_data(dataset):
    if dataset == "pride":
        return df_pride
    elif dataset == "peugeot206":
        return df_peugeot206

    elif dataset == "peugeot405":
        return df_peugeot405
    else:
        raise ValueError(
            "Data type incorrectly specified. Please choose an existing dataset."
        )


def generate_information(df):
    return len(df.index), len(df[df['Label'] == 1].index), len(df[df['Label'] == 0].index), df["price"].mean()


drc = importlib.import_module("utils.dash_reusable_components")
figs = importlib.import_module("utils.figures")

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.config['suppress_callback_exceptions'] = True
server = app.server
column_names = ['mileage', 'price', 'year']
df_pride = pd.read_excel(open('Divar Split.xlsx', 'rb'), sheet_name='pride')
df_peugeot206 = pd.read_excel(
    open('Divar Split.xlsx', 'rb'), sheet_name='peugeot 206')
df_peugeot405 = pd.read_excel(
    open('Divar Split.xlsx', 'rb'), sheet_name='peugeot 405')

df = df_peugeot405

info_405 = generate_information(df_peugeot405)
info_206 = generate_information(df_peugeot206)
info_pride = generate_information(df_pride)

app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        dcc.Store(id='dataset'),
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Divar Dashboard",
                                    href="https://github.com/amirdeljouyi/divar-dashboard",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.Div(
                            children=[
                                drc.NamedDropdown(
                                    name="Select Dataset",
                                    id="dropdown-select-dataset",
                                    options=[
                                        {"label": "Peugeot 405",
                                         "value": "peugeot405"},
                                        {
                                            "label": "Peugeot 206",
                                            "value": "peugeot206",
                                        },
                                        {
                                            "label": "Pride",
                                            "value": "pride",
                                        },
                                    ],
                                    clearable=False,
                                    searchable=False,
                                    value="peugeot405",
                                ),
                            ],
                            style={
                                "float": "right",
                                "width": "200px "
                            }
                        )
                    ],
                ),
            ],
        ),
        html.Div(
            id="tabs",
            className="tabs",
            children=[
                dcc.Tabs(
                    id="app-tab",
                    value="tab-1",
                    className="custom-tabs",
                    children=[
                        dcc.Tab(
                            id="general-tab",
                            label="General Information",
                            value="tab-1",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            id="classification-tab",
                            label="Classification & Clustering",
                            value="tab-2",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        ),
                        dcc.Tab(
                            id="query-tab",
                            label="Query",
                            value="tab-3",
                            className="custom-tab",
                            selected_className="custom-tab--selected",
                        )
                    ]
                )
            ]
        ),
        html.Div(id="body")
    ]
)

@app.callback(Output('dataset', 'data'),
              [Input("dropdown-select-dataset", "value")])
def select_datase(dropdown_select_dataset):
    return generate_data(dropdown_select_dataset).to_dict('records')

@app.callback(Output('body', 'children'),
              [Input('app-tab', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(
            children=[
                dcc.Store(id="aggregate_data"),
                html.H3("Peugout 405"),
                html.Div(
                    children=[
                        html.Div(
                            [html.H6(info_405[0]),
                             html.P("No. Cars")],
                            className="mini_container together",
                        ),
                        html.Div(
                            [html.H6(info_405[1]),
                             html.P("No. Fraud")],
                            className="mini_container together",
                        ),
                        html.Div(
                            [html.H6(info_405[2]),
                             html.P("No. Sincerity")],
                            className="mini_container together",
                        ),
                        html.Div(
                            [html.H6(info_405[3]),
                             html.P("Average Price")],
                            className="mini_container together",
                        ),
                    ],
                    id="info-container",
                    className="row container-display",
                ),
                html.H3("Peugout 206"),
                html.Div(
                    children=[
                        html.Div(
                            [html.H6(info_206[0]),
                             html.P("No. Cars")],
                            className="mini_container together",
                        ),
                        html.Div(
                            [html.H6(info_206[1]),
                             html.P("No. Fraud")],
                            className="mini_container together",
                        ),
                        html.Div(
                            [html.H6(info_206[2]),
                             html.P("No. Sincerity")],
                            className="mini_container together",
                        ),
                        html.Div(
                            [html.H6(info_206[3]),
                             html.P("Average Price")],
                            className="mini_container together",
                        ),
                    ],
                    id="info-container",
                    className="row container-display",
                ),
                html.H3("Pride"),
                html.Div(
                    children=[
                        html.Div(
                            [html.H6(info_pride[0]),
                             html.P("No. Cars")],
                            className="mini_container together",
                        ),
                        html.Div(
                            [html.H6(info_pride[1]),
                             html.P("No. Fraud")],
                            className="mini_container together",
                        ),
                        html.Div(
                            [html.H6(info_pride[2]),
                             html.P("No. Sincerity")],
                            className="mini_container together",
                        ),
                        html.Div(
                            [html.H6(info_pride[3]),
                             html.P("Average Price")],
                            className="mini_container together",
                        ),
                    ],
                    id="app-container",
                    className="row container-display",
                )
            ]

        ),
    elif tab == 'tab-2':
        return html.Div(

            id="app-container",
            # className="row",
            children=[
                html.Div(
                    id="left-column",
                    className="left-column",
                    children=[
                        drc.Card(
                            id="first-card",
                            children=[
                                drc.NamedSlider(
                                    name="Number of Cluster",
                                    id="number-cluster",
                                    min=1,
                                    max=20,
                                    step=1,
                                    marks={
                                        str(i): str(i)
                                        for i in range(1, 20)
                                    },
                                    value=10,
                                )
                            ],
                        ),
                        drc.Card(
                            id="button-card",
                            children=[
                                drc.NamedDropdown(
                                    name="Outlier Method",
                                    id="dropdown-outlier-method",
                                    options=[
                                        {
                                            "label": "Zscore",
                                            "value": "zscore",
                                        },
                                        {
                                            "label": "Quantile",
                                            "value": "quantile",
                                        },
                                    ],
                                    value="zscore",
                                    clearable=False,
                                    searchable=False,
                                ),
                                drc.NamedSlider(
                                    name="ZScore Threshold",
                                    id="zscore-threshold",
                                    min=0,
                                    max=5,
                                    value=3,
                                    step=0.5,
                                ),
                                html.Button(
                                    "Reset Threshold",
                                    id="button-zero-threshold",
                                ),
                            ],
                        ),
                        drc.Card(
                            id="predict",
                            children=[
                                html.H5("Predict"),
                                dcc.Input(
                                    placeholder='Enter a Year...',
                                    type='text',
                                    value='',
                                    id='predict-year'
                                ),
                                dcc.Input(
                                    placeholder='Enter a Mileage...',
                                    type='text',
                                    value='',
                                    id='predict-mileage'
                                ),
                                dcc.Input(
                                    placeholder='Enter a Price...',
                                    type='text',
                                    value='',
                                    id='predict-price'
                                ),
                                html.Button(
                                    "Predict",
                                    id="predict-submit",
                                ),
                                html.P("Fraud/Sincerity")
                            ]
                        )
                    ],
                ),
                html.Div(id="div-graphs"),
            ],
        )
    elif tab == 'tab-3':
        return html.Div(
            id="app-container",
            children=[
                html.Div(
                    id="table-left-column",
                    className="left-column",
                    children=[
                        drc.Card(
                            id="first-card",
                            children=[
                                drc.NamedDropdown(
                                    name="Group By",
                                    id="drop-down-group-by",
                                    options=[
                                        {"label": "Year",
                                         "value": "year"},
                                        {
                                            "label": "Mileage",
                                            "value": "mileage",
                                        },
                                        {
                                            "label": "Price",
                                            "value": "price",
                                        },
                                    ],
                                    clearable=False,
                                    searchable=False,
                                    value="year",
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    id="table-right-column",
                    className="right-column",
                )
            ]
        )

@app.callback(Output('table-right-column', 'children'),
              [Input('dataset', 'data')])
def update_table(dataset):
    df = pd.DataFrame.from_dict(dataset)
    return html.Div(
        children=[
            dash_table.DataTable(
                id='datatable-interactivity',
                columns=[
                    {"name": i, "id": i, "deletable": True} for i in df.columns
                ],
                data=df.to_dict("records"),
                editable=True,
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                row_selectable="multi",
                row_deletable=True,
                selected_rows=[],
                page_action="native",
                page_current=0,
                page_size=10,
                style_table={'overflowX': 'scroll'},
            ),
            html.Div(id='datatable-interactivity-container')
        ])


@app.callback(
    Output('datatable-interactivity-container', "children"),
    [Input('datatable-interactivity', "derived_virtual_data"),
     Input('datatable-interactivity', "derived_virtual_selected_rows"),
     Input('drop-down-group-by', 'value'),
     Input('dataset', 'data'),
     ])
def update_graphs(rows, derived_virtual_selected_rows, drop_down_group_by, dataset):
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    df = pd.DataFrame.from_dict(dataset)
    dff = df if rows is None else pd.DataFrame(rows)

    columns = set(column_names)
    columns.discard(drop_down_group_by)

    colors = ['#7FDBFF' if i in derived_virtual_selected_rows else '#0074D9'
              for i in range(len(dff))]

    return [
        html.Div(
            children=[
                html.Div(
                    className='pretty_container six columns',
                    children=[
                        html.H5("Distribution of Data by {}".format(
                            drop_down_group_by)),
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="ds-{}".format(column),
                                    figure={
                                        "data": [
                                            {
                                                "x": dff[drop_down_group_by],
                                                "y": dff[column],
                                                "type": "bar",
                                                "marker": {"color": "#13c6e9"},
                                            }
                                        ],
                                        "layout": {
                                            "xaxis": {"automargin": True},
                                            "yaxis": {
                                                "automargin": True,
                                                "title": {"text": column}
                                            },
                                            "height": 250,
                                            "margin": {"t": 10, "l": 10, "r": 10},
                                            "plot_bgcolor": "#282b38",
                                            "paper_bgcolor": "#282b38",
                                            "font": {"color": "#13c6e9"},
                                        },
                                    },
                                )
                                for column in columns if column in dff
                            ]
                        )
                    ]
                ),
                html.Div(
                    className='pretty_container six columns',
                    children=[
                        html.H5("average group by {}".format(
                            drop_down_group_by)),
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="average-{}".format(column),
                                    figure={
                                        "data": [
                                            {
                                                "x": dff[drop_down_group_by],
                                                "y": dff[column],
                                                "type": "bar",
                                                "marker": {"color": "#13c6e9"},
                                                "transforms": [dict(
                                                    type='aggregate',
                                                    groups=dff[drop_down_group_by],
                                                    aggregations=[dict(
                                                        target='y', func='avg', enabled=True),
                                                    ]
                                                )]
                                            }
                                        ],
                                        "layout": {
                                            "xaxis": {"automargin": True},
                                            "yaxis": {
                                                "automargin": True,
                                                "title": {"text": column}
                                            },
                                            "height": 250,
                                            "margin": {"t": 10, "l": 10, "r": 10},
                                            "plot_bgcolor": "#282b38",
                                            "paper_bgcolor": "#282b38",
                                            "font": {"color": "#13c6e9"},
                                        },
                                    },
                                )
                                for column in columns if column in dff
                            ]
                        )
                    ]
                ),
                html.Div(
                    className='pretty_container six columns',
                    children=[
                        html.H5("min group by {}".format(drop_down_group_by)),
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="min-{}".format(column),
                                    figure={
                                        "data": [
                                            {
                                                "x": dff[drop_down_group_by],
                                                "y": dff[column],
                                                "type": "bar",
                                                "marker": {"color": "#13c6e9"},
                                                "transforms": [dict(
                                                    type='aggregate',
                                                    groups=dff[drop_down_group_by],
                                                    aggregations=[dict(
                                                        target='y', func='min', enabled=True),
                                                    ]
                                                )]
                                            }
                                        ],
                                        "layout": {
                                            "xaxis": {"automargin": True},
                                            "yaxis": {
                                                "automargin": True,
                                                "title": {"text": column}
                                            },
                                            "height": 250,
                                            "margin": {"t": 10, "l": 10, "r": 10},
                                            "plot_bgcolor": "#282b38",
                                            "paper_bgcolor": "#282b38",
                                            "font": {"color": "#13c6e9"},
                                        },
                                    },
                                )
                                for column in columns if column in dff
                            ]
                        )
                    ]
                ),
                html.Div(
                    className='pretty_container six columns',
                    children=[
                        html.H5("max group by {}".format(drop_down_group_by)),
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="max-{}".format(column),
                                    figure={
                                        "data": [
                                            {
                                                "x": dff[drop_down_group_by],
                                                "y": dff[column],
                                                "type": "bar",
                                                "marker": {"color": "#13c6e9"},
                                                "transforms": [dict(
                                                    type='aggregate',
                                                    groups=dff[drop_down_group_by],
                                                    aggregations=[dict(
                                                        target='y', func='max', enabled=True),
                                                    ]
                                                )]
                                            }
                                        ],
                                        "layout": {
                                            "xaxis": {"automargin": True},
                                            "yaxis": {
                                                "automargin": True,
                                                "title": {"text": column}
                                            },
                                            "height": 250,
                                            "margin": {"t": 10, "l": 10, "r": 10},
                                            "plot_bgcolor": "#282b38",
                                            "paper_bgcolor": "#282b38",
                                            "font": {"color": "#13c6e9"},
                                        },
                                    },
                                )
                                for column in columns if column in dff
                            ]
                        )
                    ]
                ),
                html.Div(
                    className='pretty_container six columns',
                    children=[
                        html.H5("std group by {}".format(drop_down_group_by)),
                        html.Div(
                            children=[
                                dcc.Graph(
                                    id="std-{}".format(column),
                                    figure={
                                        "data": [
                                            {
                                                "x": dff[drop_down_group_by],
                                                "y": dff[column],
                                                "type": "bar",
                                                "marker": {"color": "#13c6e9"},
                                                "transforms": [dict(
                                                    type='aggregate',
                                                    groups=dff[drop_down_group_by],
                                                    aggregations=[dict(
                                                        target='y', func='stddev', enabled=True),
                                                    ]
                                                )]
                                            }
                                        ],
                                        "layout": {
                                            "xaxis": {"automargin": True},
                                            "yaxis": {
                                                "automargin": True,
                                                "title": {"text": column}
                                            },
                                            "height": 250,
                                            "margin": {"t": 10, "l": 10, "r": 10},
                                            "plot_bgcolor": "#282b38",
                                            "paper_bgcolor": "#282b38",
                                            "font": {"color": "#13c6e9"},
                                        },
                                    },
                                )
                                for column in columns if column in dff
                            ]
                        )
                    ]
                ),
            ]
        )
    ]


@app.callback(
    Output("zscore-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
    [State("graph-sklearn-svm", "figure")],
)
def reset_threshold_center(n_clicks, figure):
    if n_clicks:
        value = 3
    else:
        value = 2
    return value


@app.callback(
    Output("div-graphs", "children"),
    [
        Input('dataset', 'data'),
        Input("dropdown-outlier-method", "value"),
        Input("zscore-threshold", "value"),
        Input("number-cluster", "value"),
        Input("predict-submit", "n_clicks"),
        Input("predict-year", "value"),
        Input("predict-mileage", "value"),
        Input("predict-price", "value")
    ],
)
def update_cc_graphs(dataset, outlier_method, threshold, number_cluster, predict_submit, predict_year, predict_mileage, predict_price):
    if predict_submit:
        print("hi First")
        # Data Pre-processing
        df = pd.DataFrame.from_dict(dataset)

        Nc = range(1, 20)
        kmeans = [KMeans(n_clusters=i) for i in Nc]
        scaled_df = NormalizeData(df, column_names)
        elbow_curve = figs.serve_elbow_curve(kmeans, scaled_df)
        kmeans = KMeans(n_clusters=number_cluster, random_state=0).fit(scaled_df)

        scaled_df = dfAfterKmeans(kmeans, scaled_df)
        swarm_plot = figs.serve_swarm_plot(scaled_df)

        outliers = outlierss(scaled_df, number_cluster, outlier_method, threshold)

        df['Label'] = df['ID'].apply(lambda x: 1 if checkTrue(
            x, outliers, number_cluster) else 0)
        scaled_df['Label'] = scaled_df['ID'].apply(
            lambda x: 1 if checkTrue(x, outliers, number_cluster) else 0)

        # pca = PCA(n_components=2)
        # principalComponents = pca.fit_transform(scaled_df.loc[:, column_names])
        # principalDf = pd.DataFrame(data=principalComponents, columns=[
        #     'principal component 1', 'principal component 2'])
        # finalDf = pd.concat([principalDf, scaled_df['Label']], axis=1)

        # trainX, testX, trainy, testy = train_test_split(finalDf.loc[:, [
        #     'principal component 1', 'principal component 2']], df.loc[:, ['Label']], test_size=0.4, random_state=2)

        trainX, testX, trainy, testy = train_test_split(
            scaled_df.loc[:, column_names], scaled_df.loc[:, ['Label']], test_size=0.4, random_state=2)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(trainX, trainy)
        probs = model.predict_proba(testX)
        probs = probs[:, 1]

        roc_figure = figs.serve_roc_curve(
             model=model, X_test=testX, y_test=testy, probs=probs)

        precision_recall_figure = figs.serve_precision_recall(
             model=model, X_test=testX, y_test=testy, probs=probs)

        print("hi")

        predict = ""

        test_predict = [[int(predict_mileage), int(
            predict_price), int(predict_year)]]
        predict = model.predict(test_predict)
        print(predict)

        return [
            # html.Div(
            #     id="svm-graph-container",
            #     children=dcc.Loading(
            #         className="graph-wrapper",
            #         children=dcc.Graph(id="graph-sklearn-svm", figure=prediction_figure),
            #         style={"display": "none"},
            #     ),
            # ),
            html.Div(
                id="graphs-container",
                children=[
                    dcc.Loading(
                        className="graph-wrapper pretty_container four columns",
                        children=dcc.Graph(id="elbow-curve", figure=elbow_curve),
                    ),
                    dcc.Loading(
                        className="graph-wrapper pretty_container eight columns",
                        children=dcc.Graph(
                            id="swarm-plot", figure=swarm_plot
                        ),
                    ),
                    dcc.Loading(
                        className="graph-wrapper pretty_container six columns",
                        children=dcc.Graph(
                            id="roc-curve", figure=roc_figure
                        ),
                    ),
                    dcc.Loading(
                        className="graph-wrapper pretty_container six columns",
                        children=dcc.Graph(
                            id="precision-recall", figure=precision_recall_figure
                        ),
                    ),
                    dcc.Loading(
                        className="graph-wrapper pretty_container six columns",
                        children=[
                            html.H5("Predict"),
                            html.H6(predict)
                        ]
                    )
                ]
            ),
        ]


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
