import base64
from dataclasses import dataclass
from pathlib import Path

from jupyter_dash import JupyterDash
from dash import html, dcc, State
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
from numpy.typing import NDArray

from sbayes.load_data import Objects, Confounder
from sbayes.results import Results
from sbayes.util import read_data_csv, compute_delaunay, gabriel_graph_from_delaunay

# Define paths

CASE_STUDY = "global"
# CASE_STUDY = "OoA"
NAME = "tmp"
# NAME = "geo:simu_size:ua_w:sym3_fam:unif"
# NAME = "server/geo:simu_size:ua_w:sym3_fam:unif"
# NAME = "server/geo:sum+sigmoid+600+300_size:ua_w:sym3_fam:unif"
# NAME = "geo:sum+sigmoid+600+300_size:ua_w:sym3_fam:unif"
# NAME = "geo:unif_size:us_w:sym3_fam:unif"
K = 4
RUN = 0


# # clusters_path = './experiments/global/results/2022-10-23_13-31/K3/clusters_K3_0.txt'
# # clusters_path = "./experiments/global/results/test/K3/clusters_K3_0.txt"
# clusters_path = f"./experiments/{CASE_STUDY}/results/{NAME}/K{K}/clusters_K{K}_{RUN}.txt"
# # clusters_path = './experiments/south_america/results/test/K3/clusters_K3_0.txt'

if CASE_STUDY == "global":
    # data_path = "./experiments/global/data/features.csv"
    data_path = "./experiments/global/data/grambank-original-for-sbayes.csv"
elif CASE_STUDY == "OoA":
    data_path = "./experiments/OoA/data/america.csv"
elif CASE_STUDY == "south_america":
    data_path = "./experiments/south_america/data/features.csv"
else:
    raise ValueError("Provide data_path")


# Some convenience functions


def read_families(data_path: str) -> Confounder:
    data = read_data_csv(data_path)
    return Confounder.from_dataframe(data, confounder_name="family")


def read_objects(data_path: str) -> Objects:
    print("Reading input data...")
    data = read_data_csv(data_path)
    return Objects.from_dataframe(data)


def reproject_locations(locations, data_proj, map_proj):
    if data_proj == map_proj:
        return locations
    loc = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*locations.T), crs=data_proj)
    loc_re = loc.to_crs(map_proj).geometry
    return np.array([loc_re.x, loc_re.y]).T


def min_and_max_with_padding(x, pad=0.05):
    lower = np.min(x)
    upper = np.max(x)
    diff = upper - lower
    return lower - pad * diff, upper + pad * diff


object_size: int = 10
# data_projection: str = "+proj=eqdc +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs "
data_projection: str = "EPSG:4326"


def find_biggest_angle_gap(degrees: NDArray[float]) -> float:
    degrees = np.sort(degrees)
    np.append(degrees, degrees[0] + 360)
    i = np.argmax(np.diff(degrees))
    return (degrees[i+1] + degrees[i]) / 2


def cluster_to_graph(locations):
    delaunay = compute_delaunay(locations)
    graph_connections = gabriel_graph_from_delaunay(delaunay, locations)

    x, y = [], []
    for i1, i2 in graph_connections:
        x += [locations[i1, 0], locations[i2, 0], None]
        y += [locations[i1, 1], locations[i2, 1], None]
    return x, y


# Load data
objects = read_objects(data_path)
families = read_families(data_path)
locations = reproject_locations(objects.locations, data_projection, "EPSG:4326")

family_names = np.array(families.group_names + [""])
family_ids = []
for i, lang in enumerate(objects.names):
    i_fam = np.flatnonzero(families.group_assignment[:, i])
    i_fam = i_fam[0] if len(i_fam) > 0 else families.n_groups
    family_ids.append(i_fam)
family_ids = np.array(family_ids)

object_data = pd.DataFrame(
    {
        "x": locations[:, 0],
        "y": locations[:, 1],
        "name": objects.names,
        "family": family_names[family_ids],
    }
)


@dataclass
class AppState:

    clusters_path = None
    _clusters = None
    fig = None
    lines = None
    scatter = None
    cluster_colors = None

    @property
    def clusters(self):
        return self._clusters

    @clusters.setter
    def clusters(self, clusters):
        self._clusters = clusters
        self.cluster_colors = self.get_cluster_colors(self.n_clusters)

    @staticmethod
    def get_cluster_colors(K):
        # cm = plt.get_cmap('gist_rainbow')
        # cluster_colors = [colors.to_hex(c) for c in cm(np.linspace(0, 1, K, endpoint=False))]
        colors = []
        for i, x in enumerate(np.linspace(0, 1, K, endpoint=False)):
            b = i % 2
            h = x % 1
            s = 0.6 + 0.4 * b
            v = 0.5 + 0.3 * (1 - b)
            colors.append(
                mpl_colors.to_hex(mpl_colors.hsv_to_rgb((h, s, v)))
            )
        return colors

    @property
    def n_clusters(self) -> int:
        return self.clusters.shape[0]

    @property
    def n_samples(self) -> int:
        return self.clusters.shape[1]

# Initialized app
app = JupyterDash(__name__)
state = AppState()

upload_box_style = {
    "width": "100%",
    "height": "60px",
    "lineHeight": "60px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "5px",
    "textAlign": "center",
    "margin": "10px",
}

# Set up the layout
app.layout = html.Div(
    children=[
        dcc.Upload(
            id='upload-clusters',
            children=html.Div([
                'Drag and drop or select the ', html.B('clusters file')
            ]),
            style=upload_box_style,
        ),
        html.Div(id='uploaded'),
    ]
)


@app.callback(Output('uploaded', 'children'),
              Input('upload-clusters', 'contents'),
              Input('upload-clusters', 'filename'),
              State('upload-clusters', 'last_modified')
              )
def update_output(content, filename, list_of_dates):
    if content is None:
        return

    content_type, content_bytestr = content.split(',')
    clusters_str = str(base64.b64decode(content_bytestr))[2:-1]
    clusters_str = clusters_str.replace(r"\t", "\t").replace(r"\n", "\n")

    clusters_file_name = Path(filename)
    state.clusters_path = clusters_file_name

    # locations = reproject_locations(objects.locations, data_projection, "+proj=eqdc +lat_0=-32 +lon_0=-60 +lat_1=-5 +lat_2=-42 +x_0=0 +y_0=0 +ellps=aust_SA +units=m +no_defs")
    cut_longitude = find_biggest_angle_gap(locations[:, 0])
    locations[:, 0] = (locations[:, 0] - cut_longitude) % 360 + cut_longitude

    fig = px.scatter_geo(
        object_data,
        lat="y", lon="x",
        hover_data=["name", "family"],
        projection="natural earth",
    )

    # Load data
    clusters = Results.read_clusters_from_str(clusters_str)
    n_clusters, n_samples, n_sites = clusters.shape

    for i in range(n_clusters):
        fig_lines = px.line_geo(lat=[None], lon=[None])
        fig = go.Figure(fig.data + fig_lines.data)

    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        geo=dict(
            lonaxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[*min_and_max_with_padding(locations[:, 0])],
                dtick=5,
            ),
            lataxis=dict(
                showgrid=True,
                gridwidth=0.5,
                range=[*min_and_max_with_padding(locations[:, 1])],
                dtick=5,
            ),
        ),
    )

    state.clusters = clusters

    state.lines = fig.data[1:]
    state.scatter = fig.data[0]

    # Fix z-order so that lines are behind scatter:
    fig.data = fig.data[::-1]

    # for i in range(n_clusters):
    #     f = fig.add_trace(
    #         go.Scatter(x=[np.nan], y=[np.nan], legendgroup=f"Cluster {i}", marker_color=cluster_colors[i], name=f"Cluster {i}", visible="legendonly")
    #     )

    fig.update_layout(showlegend=True)
    state.fig = fig

    return html.Div([
        html.P(id="sample", children="Sample number"),
        dcc.Slider(id="i_sample", value=0, step=1, min=0, max=state.n_samples-1,
                   marks={i:i for i in range(0, state.n_samples, max(1, state.n_samples//10))}),
        dcc.Graph(id="map"),
    ])


@app.callback(
    Output("map", "figure"),
    Input("i_sample", "value"),
)
def update_map(i_sample: int):
    if state.clusters_path is None:
        return None

    colors = np.full(objects.n_objects, "lightgrey", dtype=object)
    for i, c in enumerate(state.clusters[:, i_sample, :]):
        print(c.shape)
        print(locations.shape)
        state.lines[i].lon, state.lines[i].lat = cluster_to_graph(locations[c])
        colors[c] = state.cluster_colors[i]
        state.lines[i].line.color = state.cluster_colors[i]

    state.scatter.marker.color = list(colors)
    return state.fig


app.run_server()
