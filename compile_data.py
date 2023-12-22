"""

ariel@oapd
22/12/2023

Compiles data in a table of integrated values

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pysinopsis.output import SinopsisCube
from pysinopsis.utils import gini
import pandas as pd
import pickle

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Asymetries/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')
galaxy_list = sample['ID']

problematic_cases = np.genfromtxt(data_dir + 'mass_formed_issues.txt', dtype=str)

mfraction_maps = pickle.load(open(data_dir + 'mass_formed.pkl', 'rb'))

sample_properties = {}
sample_properties['galaxy'] = galaxy_list
sample_properties['bad_galaxy'] = np.zeros(shape=len(galaxy_list), dtype=bool)
sample_properties['n_spaxels'] = np.empty(shape=len(galaxy_list))
sample_properties['n_disk'] = np.empty(shape=len(galaxy_list))
sample_properties['n_spaxels_disk'] = np.empty(shape=len(galaxy_list))
sample_properties['flux_spaxels'] = np.empty(shape=len(galaxy_list))
sample_properties['flux_disk'] = np.empty(shape=len(galaxy_list))
sample_properties['flux_spaxels_disk'] = np.empty(shape=len(galaxy_list))
sample_properties['mass'] = np.empty(shape=len(galaxy_list))
sample_properties['mass_corr'] = np.empty(shape=len(galaxy_list))
sample_properties['mean_mass_fraction'] = np.empty(shape=len(galaxy_list))
sample_properties['median_mass_fraction'] = np.empty(shape=len(galaxy_list))
sample_properties['std_mass_fraction'] = np.empty(shape=len(galaxy_list))
sample_properties['gini_mass_fraction'] = np.empty(shape=len(galaxy_list))


for galaxy in galaxy_list:

    galaxy_index = np.argwhere(galaxy_list == galaxy)[0][0]

    if galaxy in problematic_cases:
        sample_properties['bad_galaxy'][galaxy_index] = True
        continue

    galaxy_path = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/' + galaxy + '/'

    sinopsis_cube = SinopsisCube(galaxy_path)   

    contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
    g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

    sinopsis_flag = sinopsis_cube.mask == 1
    contours_flag = contours > 2
    contours_and_sinopsis_flag = contours_flag & sinopsis_flag

    n_spaxels = sinopsis_flag.sum()
    n_disk = contours_flag.sum()
    n_spaxels_disk = contours_and_sinopsis_flag.sum()
    flux_spaxels = np.ma.masked_array(g_band, mask=~sinopsis_flag).sum()
    flux_disk = np.ma.masked_array(g_band, mask=~contours_flag).sum()
    flux_spaxels_disk = np.ma.masked_array(g_band, mask=~contours_and_sinopsis_flag).sum()
      
    sample_properties['n_spaxels'][galaxy_index] = n_spaxels
    sample_properties['n_disk'][galaxy_index] = n_disk
    sample_properties['n_spaxels_disk'][galaxy_index] = n_spaxels_disk
    sample_properties['flux_spaxels'][galaxy_index] = flux_spaxels
    sample_properties['flux_disk'][galaxy_index] = flux_disk
    sample_properties['flux_spaxels_disk'][galaxy_index] = flux_spaxels_disk
    sample_properties['mass'][galaxy_index] = np.sum(sinopsis_cube.properties['TotMass2'])
    sample_properties['mass_corr'][galaxy_index] = np.sum(sinopsis_cube.properties['TotMass2']) * (flux_disk/flux_spaxels_disk)
    sample_properties['mean_mass_fraction'][galaxy_index] = np.mean(mfraction_maps[galaxy]['m1'][contours_and_sinopsis_flag])
    sample_properties['median_mass_fraction'][galaxy_index] = np.median(mfraction_maps[galaxy]['m1'][contours_and_sinopsis_flag])
    sample_properties['std_mass_fraction'][galaxy_index] = np.std(mfraction_maps[galaxy]['m1'][contours_and_sinopsis_flag])
    sample_properties['gini_mass_fraction'][galaxy_index] = gini(mfraction_maps[galaxy]['m1'][contours_and_sinopsis_flag].ravel())


sample_properties_df = pd.DataFrame(sample_properties)

sample_properties_df.to_csv(data_dir + 'sample_properties.csv', index=False)