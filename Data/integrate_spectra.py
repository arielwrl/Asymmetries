import numpy as np
from astropy.io import fits
import itertools
from pysinopsis.output import SinopsisCube
from pysinopsis.utils import box_filter
import pandas as pd

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Data/dashboard/data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'
plots_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB regions/Plots/maps_and_centers/'

sample_all = pd.read_csv(data_dir + 'all_galaxy_sample.csv')


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

    return wl, integrated_spectrum


for galaxy in sample_all['ID']:

    print(galaxy)

    wl15, flux15 = calc_integrated_spectra(galaxy, sigma=1.5)
    wl3, flux3 = calc_integrated_spectra(galaxy, sigma=3)
    wl5, flux5 = calc_integrated_spectra(galaxy, sigma=5)

    labels = [np.full_like(wl15, '1.5'), np.full_like(wl15, '3'), np.full_like(wl15, '5')]
    labels = np.concatenate(labels)         
    wl_all = np.concatenate([wl15, wl15, wl15])
    flux_all = np.concatenate([flux15, flux3, flux5])

    df = pd.DataFrame({'Wavelength': wl_all, 'Flux': flux_all, 'Contour Limit': labels})

    df.to_csv(data_dir + 'dashboard_data/' + galaxy + '.csv')


for galaxy in sample_all['ID']:

    print(galaxy)

    wl15, flux15 = calc_integrated_spectra(galaxy, sigma=1.5)
    wl3, flux3 = calc_integrated_spectra(galaxy, sigma=3)
    wl5, flux5 = calc_integrated_spectra(galaxy, sigma=5)

    wl15, flux15 = box_filter(wl15, flux15, box_width=6)
    wl3, flux3 = box_filter(wl3, flux3, box_width=6)
    wl5, flux5 = box_filter(wl5, flux5, box_width=6)

    labels = [np.full_like(wl15, '1.5'), np.full_like(wl15, '3'), np.full_like(wl15, '5')]
    labels = np.concatenate(labels)         
    wl_all = np.concatenate([wl15, wl15, wl15])
    flux_all = np.concatenate([flux15, flux3, flux5])

    df = pd.DataFrame({'Wavelength': wl_all, 'Flux': flux_all, 'Contour Limit': labels})

    df.to_csv(data_dir + 'dashboard_data/' + galaxy + '_filtered.csv')



for galaxy in sample_all['ID']:

    wl15, flux15 = calc_integrated_spectra(galaxy, sigma=1.5, restframe=True)
    wl3, flux3 = calc_integrated_spectra(galaxy, sigma=3, restframe=True)
    wl5, flux5 = calc_integrated_spectra(galaxy, sigma=5, restframe=True)

    labels = [np.full_like(wl15, '1.5'), np.full_like(wl15, '3'), np.full_like(wl15, '5')]
    labels = np.concatenate(labels)
    wl_all = np.concatenate([wl15, wl15, wl15])
    flux_all = np.concatenate([flux15, flux3, flux5])

    df = pd.DataFrame({'Wavelength': wl_all, 'Flux': flux_all, 'Contour Limit': labels})

    df = pd.DataFrame({'Wavelength': wl_all, 'Flux': flux_all, 'Contour Limit': labels})

    df.to_csv(data_dir + 'dashboard_data/' + galaxy + '_restframe.csv')


for galaxy in sample_all['ID']:

    wl15, flux15 = calc_integrated_spectra(galaxy, sigma=1.5, restframe=True)
    wl3, flux3 = calc_integrated_spectra(galaxy, sigma=3, restframe=True)
    wl5, flux5 = calc_integrated_spectra(galaxy, sigma=5, restframe=True)

    wl15, flux15 = box_filter(wl15, flux15, box_width=6)
    wl3, flux3 = box_filter(wl3, flux3, box_width=6)
    wl5, flux5 = box_filter(wl5, flux5, box_width=6)

    labels = [np.full_like(wl15, '1.5'), np.full_like(wl15, '3'), np.full_like(wl15, '5')]
    labels = np.concatenate(labels)
    wl_all = np.concatenate([wl15, wl15, wl15])
    flux_all = np.concatenate([flux15, flux3, flux5])

    df = pd.DataFrame({'Wavelength': wl_all, 'Flux': flux_all, 'Contour Limit': labels})

    df = pd.DataFrame({'Wavelength': wl_all, 'Flux': flux_all, 'Contour Limit': labels})

    df.to_csv(data_dir + 'dashboard_data/' + galaxy + '_restframe_filtered.csv')
