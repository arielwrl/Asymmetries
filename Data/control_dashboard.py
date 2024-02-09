from dash import Dash, dcc, html, Input, Output, dash_table
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn import metrics, datasets
import plotly.express as px
import dash_bootstrap_components as dbc

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import itertools
from pysinopsis.output import SinopsisCube
from pysinopsis.utils import calc_center_of_mass, box_filter
import pandas as pd
from scipy.ndimage import gaussian_filter
import seaborn as sns
from toolbox.kubevizResults import kubevizResults as kv
import plotly.io as pio

pio.templates.default = 'plotly_white'

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'
plots_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB regions/Plots/maps_and_centers/'

sample_all = pd.read_csv(data_dir + 'all_galaxy_sample.csv')

reference_lines = {'OII': 3727, 'H\u03B4': 4101, 'H\u03B2': 4861, 'OIII': 5007, 'H\u03B1': 6563}

app = Dash(external_stylesheets=[dbc.themes.LUX])

controls = dbc.Card(
    [
        html.Div(
            [
                html.H4("Select Galaxy:"),
                dcc.Dropdown(id="dropdown", options=sample_all['ID'], value="A370_01", clearable=False),
            ]
        ),
        html.Div(
            [
                dbc.Label("Box Filter Width"),
                dcc.Slider(1, 20, 1, marks={1: '1', 5: '5', 10: '10', 15: '15', 20: '20'}, value=1, id='width_slider'),
            ]
        ),
        html.Div(
            [
                html.H4(id='data_summary_title'),
                html.Div(id='redshift'),
                html.Div(id="category"),
                html.Div(id="sinopsis_fraction"),
                html.Div(id="mag_disk"),
                html.Div(id="sn"),
                html.Div(id="nad_notch")
            ]
        ),
    ],
    body=True,
)

spectra = dbc.Card(html.Div([dcc.Graph(id="observed_spectrum"), 
                             dcc.Graph(id="restframe_spectrum")]))

app.layout = dbc.Container(
    [
        html.H1("Checking Intermediate-z Sample Galaxies"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(spectra, md=8),
            ],
            align="top",
        ),
    ],
    fluid=True,
)


def calc_integrated_spectra(galaxy, sigma=5, restframe=False):

    galaxy_index = np.argwhere(sample_all['ID'] == galaxy)[0][0]

    galaxy_path = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/' + galaxy + '/'

    sinopsis_cube = SinopsisCube(galaxy_path)   

    contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data

    wl = sinopsis_cube.wl
    if restframe:
        wl /= (1 + sample_all['z'][galaxy_index])

    contours_flag = contours >= sigma

    spec_list = []

    for i, j in itertools.product(range(sinopsis_cube.cube_shape[1]), range(sinopsis_cube.cube_shape[2])):

        if contours_flag[i, j] == False:
            continue

        flux = sinopsis_cube.f_obs[:, i, j]

        spec_list.append(flux)

    integrated_spectrum = np.sum(spec_list, axis=0)
    if restframe:
        integrated_spectrum *= (1 + sample_all['z'][galaxy_index])

    integrated_spectra = {'wl': wl,
                          'integrated_spectrum': integrated_spectrum,
                          'z': sample_all['z'][galaxy_index]}

    return integrated_spectra


@app.callback(
    Output("data_summary_title", "children"),
    Output("redshift", "children"),
    Output("category", "children"),
    Output("sinopsis_fraction", "children"),
    Output("mag_disk", "children"),
    Output("sn", "children"),
    Output("nad_notch", "children"),
    Input("dropdown", "value"),
)


def get_galaxy_table(galaxy):

    galaxy_data = sample_all[sample_all['ID']==galaxy]

    return 'This is the data for ' + galaxy_data['ID'], 'Redshift: {z}'.format(z=galaxy_data['z'].values[0]), \
           'Category: ' + str(galaxy_data['Category'].values[0]), 'Sinopsis Fraction: {fraction:.2f}'.format(fraction=galaxy_data['sinopsis_fraction'].values[0]), \
           'g-band magnitude: {mag:.2f}'.format(mag=galaxy_data['mag_disk'].values[0]), 'S/N (5635): {sn:.2f}'.format(sn=galaxy_data['sn_4020_int'].values[0]), \
           'NaD notch: ' + str(galaxy_data['nad_notch'].values[0]), 
            

@app.callback(
    Output("observed_spectrum", "figure"),
    Input("dropdown", "value"),
    Input("width_slider", "value"),
)


def plot_integrated_spectra(galaxy, box_width):

    spectra15 = calc_integrated_spectra(galaxy, sigma=1.5)
    spectra3 = calc_integrated_spectra(galaxy, sigma=3)
    spectra5 = calc_integrated_spectra(galaxy, sigma=5)

    redshift = spectra15['z']

    wl15, flux15 = spectra15['wl'], spectra15['integrated_spectrum']
    wl3, flux3 = spectra3['wl'], spectra3['integrated_spectrum']
    wl5, flux5 = spectra5['wl'], spectra5['integrated_spectrum']

    wl_plot, flux_plot15 = box_filter(wl15, flux15, box_width)
    flux_plot3 = box_filter(wl3, flux3, box_width)[1]
    flux_plot5 = box_filter(wl5, flux5, box_width)[1]

    labels = [np.full_like(wl_plot, '1.5'), np.full_like(wl_plot, '3'), np.full_like(wl_plot, '5')]
    labels = np.concatenate(labels)
    wl_all = np.concatenate([wl_plot, wl_plot, wl_plot])
    flux_all = np.concatenate([flux_plot15, flux_plot3, flux_plot5])

    plot_df = pd.DataFrame({'Wavelength': wl_all, 'Flux': flux_all, 'Contour Limit': labels})

    fig = px.line(data_frame=plot_df,
                  x='Wavelength',
                  y='Flux',
                  color='Contour Limit',
        title='Integrated Spectrum (Observed Frame)',
    )

    for line in reference_lines.keys():
        fig.add_vline(x=reference_lines[line] * (1 + redshift), line_dash='dash', annotation_text=line, annotation_position='top right', line_width=0.75)

    fig.add_vrect(x0=(5635-50) * (1 + redshift), x1=(5635+50) * (1 + redshift), 
                annotation_text="S/N", annotation_position="top",
                fillcolor="green", opacity=0.25, line_width=0, col=1)
    
    if galaxy.split('_')[0] in ['MACS0257', 'RXJ1347', 'SMACS2131']:
        fig.add_vrect(x0=5760, x1=6010, annotation_text="NaD notch", annotation_position="top",
                      fillcolor="red", opacity=0.25, line_width=0)

    return fig


@app.callback(
    Output("restframe_spectrum", "figure"),
    Input("dropdown", "value"),
    Input("width_slider", "value"),
)


def plot_integrated_spectra_restframe(galaxy, box_width):

    spectra15 = calc_integrated_spectra(galaxy, sigma=1.5, restframe=True)
    spectra3 = calc_integrated_spectra(galaxy, sigma=3, restframe=True)
    spectra5 = calc_integrated_spectra(galaxy, sigma=5, restframe=True)

    redshift = spectra15['z']

    wl15, flux15 = spectra15['wl'], spectra15['integrated_spectrum']
    wl3, flux3 = spectra3['wl'], spectra3['integrated_spectrum']
    wl5, flux5 = spectra5['wl'], spectra5['integrated_spectrum']

    wl_plot, flux_plot15 = box_filter(wl15, flux15, box_width)
    flux_plot3 = box_filter(wl3, flux3, box_width)[1]
    flux_plot5 = box_filter(wl5, flux5, box_width)[1]

    labels = [np.full_like(wl_plot, '1.5'), np.full_like(wl_plot, '3'), np.full_like(wl_plot, '5')]
    labels = np.concatenate(labels)
    wl_all = np.concatenate([wl_plot, wl_plot, wl_plot])
    flux_all = np.concatenate([flux_plot15, flux_plot3, flux_plot5])

    plot_df = pd.DataFrame({'Wavelength': wl_all, 'Flux': flux_all, 'Contour Limit': labels})

    fig = px.line(data_frame=plot_df,
                  x='Wavelength',
                  y='Flux',
                  color='Contour Limit',
        title='Integrated Spectrum (Rest Frame)',
    )

    for line in reference_lines.keys():
        fig.add_vline(x=reference_lines[line], line_dash='dash', annotation_text=line, annotation_position='top right', line_width=0.75)

    fig.add_vrect(x0=5635-50, x1=5635+50, 
                  annotation_text="S/N", annotation_position="top",
                  fillcolor="green", opacity=0.25, line_width=0, col=1)

    if galaxy.split('_')[0] in ['MACS0257', 'RXJ1347', 'SMACS2131']:
        fig.add_vrect(x0=5760 / (1 + redshift), x1=6010 / (1 + redshift), 
                      annotation_text="NaD notch", annotation_position="top",
                      fillcolor="red", opacity=0.25, line_width=0)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)