import time
import importlib

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State


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


drc = importlib.import_module("utils.dash_reusable_components")
figs = importlib.import_module("utils.figures")

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
server = app.server
column_names = ['mileage','price','year']
df_pride = pd.read_excel(open('Divar Split.xlsx', 'rb'),sheet_name='pride')
df_peugeot206 = pd.read_excel(open('Divar Split.xlsx', 'rb'),sheet_name='peugeot 206')
df_peugeot405 =pd.read_excel(open('Divar Split.xlsx', 'rb'),sheet_name='peugeot 405')

def dfAfterKmeans(kmeans, df):
    dist = kmeans.transform(df)**2
    df['sqdist'] = dist.sum(axis=1).round(2)
    df['cluster'] = kmeans.labels_
    df['ID'] = np.arange(len(df))
    return df

def NormalizeData(df,column_names):
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


app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
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
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select Dataset",
                                            id="dropdown-select-dataset",
                                            options=[
                                                {"label": "Peugeot 405", "value": "peugeot405"},
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
                                        drc.NamedSlider(
                                            name="Number of Cluster",
                                            id="number-cluster",
                                            min=1,
                                            max=20,
                                            step=1,
                                            marks={
                                                str(i): str(i)
                                                for i in range(1,20)
                                            },
                                            value=10,
                                        ),
                                        drc.NamedSlider(
                                            name="Noise Level",
                                            id="slider-dataset-noise-level",
                                            min=0,
                                            max=1,
                                            marks={
                                                i / 10: str(i / 10)
                                                for i in range(0, 11, 2)
                                            },
                                            step=0.1,
                                            value=0.2,
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Outlier Method",
                                            id="outlier-method",
                                            options=[
                                                {
                                                    "label": "Radial basis function (RBF)",
                                                    "value": "rbf",
                                                },
                                                {"label": "Linear", "value": "linear"},
                                                {
                                                    "label": "Polynomial",
                                                    "value": "poly",
                                                },
                                                {
                                                    "label": "Sigmoid",
                                                    "value": "sigmoid",
                                                },
                                            ],
                                            value="rbf",
                                            clearable=False,
                                            searchable=False,
                                        ),
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-threshold",
                                            min=0,
                                            max=1,
                                            value=0.5,
                                            step=0.01,
                                        ),
                                        html.Button(
                                            "Reset Threshold",
                                            id="button-zero-threshold",
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="last-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Kernel",
                                            id="dropdown-svm-parameter-kernel",
                                            options=[
                                                {
                                                    "label": "Radial basis function (RBF)",
                                                    "value": "rbf",
                                                },
                                                {"label": "Linear", "value": "linear"},
                                                {
                                                    "label": "Polynomial",
                                                    "value": "poly",
                                                },
                                                {
                                                    "label": "Sigmoid",
                                                    "value": "sigmoid",
                                                },
                                            ],
                                            value="rbf",
                                            clearable=False,
                                            searchable=False,
                                        ),
                                        drc.NamedSlider(
                                            name="Cost (C)",
                                            id="slider-svm-parameter-C-power",
                                            min=-2,
                                            max=4,
                                            value=0,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-2, 5)
                                            },
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-C-coef",
                                            min=1,
                                            max=9,
                                            value=1,
                                        ),
                                        drc.NamedSlider(
                                            name="Degree",
                                            id="slider-svm-parameter-degree",
                                            min=2,
                                            max=10,
                                            value=3,
                                            step=1,
                                            marks={
                                                str(i): str(i) for i in range(2, 11, 2)
                                            },
                                        ),
                                        drc.NamedSlider(
                                            name="Gamma",
                                            id="slider-svm-parameter-gamma-power",
                                            min=-5,
                                            max=0,
                                            value=-1,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-5, 1)
                                            },
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-gamma-coef",
                                            min=1,
                                            max=9,
                                            value=5,
                                        ),
                                        html.Div(
                                            id="shrinking-container",
                                            children=[
                                                html.P(children="Shrinking"),
                                                dcc.RadioItems(
                                                    id="radio-svm-parameter-shrinking",
                                                    labelStyle={
                                                        "margin-right": "7px",
                                                        "display": "inline-block",
                                                    },
                                                    options=[
                                                        {
                                                            "label": " Enabled",
                                                            "value": "True",
                                                        },
                                                        {
                                                            "label": " Disabled",
                                                            "value": "False",
                                                        },
                                                    ],
                                                    value="True",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "marks"),
    [Input("slider-svm-parameter-gamma-power", "value")],
)
def update_slider_svm_parameter_gamma_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-svm-parameter-C-coef", "marks"),
    [Input("slider-svm-parameter-C-power", "value")],
)
def update_slider_svm_parameter_C_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
    [State("graph-sklearn-svm", "figure")],
)
def reset_threshold_center(n_clicks, figure):
    if n_clicks:
        Z = np.array(figure["data"][0]["z"])
        value = -Z.min() / (Z.max() - Z.min())
    else:
        value = 0.4959986285375595
    return value


# Disable Sliders if kernel not in the given list
@app.callback(
    Output("slider-svm-parameter-degree", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_degree(kernel):
    return kernel != "poly"


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_coef(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("slider-svm-parameter-gamma-power", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_power(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("div-graphs", "children"),
    [
        Input("dropdown-svm-parameter-kernel", "value"),
        Input("slider-svm-parameter-degree", "value"),
        Input("slider-svm-parameter-C-coef", "value"),
        Input("slider-svm-parameter-C-power", "value"),
        Input("slider-svm-parameter-gamma-coef", "value"),
        Input("slider-svm-parameter-gamma-power", "value"),
        Input("dropdown-select-dataset", "value"),
        Input("slider-dataset-noise-level", "value"),
        Input("radio-svm-parameter-shrinking", "value"),
        Input("slider-threshold", "value"),
        Input("cluster-number", "value"),
    ],
)
def update_svm_graph(
    kernel,
    degree,
    C_coef,
    C_power,
    gamma_coef,
    gamma_power,
    dataset,
    noise,
    shrinking,
    threshold,
    number_cluster,
):
    t_start = time.time()
    h = 0.3  # step size in the mesh

    # Data Pre-processing
    df = generate_data(dataset=dataset)

    Nc = range(1, 20)
    kmeans= [KMeans(n_clusters=i) for i in Nc]
    scaled_df = NormalizeData(df,column_names)
    elbow_curve = figs.serve_elbow_curve(kmeans, scaled_df)
    kmeans= KMeans(n_clusters = number_cluster, random_state=0).fit(scaled_df_pride)

    scaled_df = dfAfterKmeans(kmeans, scaled_df)
    swarm_plot = figs.serve_swarm_plot(scaled_df)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.4, random_state=42
    # )

    # x_min = X[:, 0].min() - 0.5
    # x_max = X[:, 0].max() + 0.5
    # y_min = X[:, 1].min() - 0.5
    # y_max = X[:, 1].max() + 0.5
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # C = C_coef * 10 ** C_power
    # gamma = gamma_coef * 10 ** gamma_power

    # if shrinking == "True":
    #     flag = True
    # else:
    #     flag = False

    # Train SVM
    # clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, shrinking=flag)
    # clf.fit(X_train, y_train)

    # # Plot the decision boundary. For that, we will assign a color to each
    # # point in the mesh [x_min, x_max]x[y_min, y_max].
    # if hasattr(clf, "decision_function"):
    #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # else:
    #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # prediction_figure = figs.serve_prediction_plot(
    #     model=clf,
    #     X_train=X_train,
    #     X_test=X_test,
    #     y_train=y_train,
    #     y_test=y_test,
    #     Z=Z,
    #     xx=xx,
    #     yy=yy,
    #     mesh_step=h,
    #     threshold=threshold,
    # )

    # roc_figure = figs.serve_roc_curve(model=clf, X_test=X_test, y_test=y_test)

    # confusion_figure = figs.serve_pie_confusion_matrix(
    #     model=clf, X_test=X_test, y_test=y_test, Z=Z, threshold=threshold
    # )

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
            children = [dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="elbow-curve", figure=elbow_curve),
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="swarm-plot", figure=swarm_plot
                    ),
                ),
            ]
        ),
    ]


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
