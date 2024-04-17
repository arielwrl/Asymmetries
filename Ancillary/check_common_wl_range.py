import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pysinopsis.output import SinopsisCube
from pysinopsis.utils import gini
import pandas as pd
import pickle
import itertools

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')
galaxy_list = sample['ID']

select_list = galaxy_list[(sample['Category'] == 'Jellyfish') | (sample['Category'] == 'Post-Starburst') | (sample['Category'] == 'Cluster Control') | (sample['Category'] == 'Field Control')]

problematic_cases = np.genfromtxt(data_dir + 'mass_formed_issues.txt', dtype=str)

wl_list = []
galaxy_id_list = []

for galaxy in select_list:

    galaxy_index = np.argwhere(galaxy_list == galaxy)[0][0]

    if galaxy in problematic_cases:
        continue

    galaxy_path = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/' + galaxy + '/'

    sinopsis_cube = SinopsisCube(galaxy_path)   

    contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
    g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

    sinopsis_flag = sinopsis_cube.mask == 1
    contours_flag = contours > 3
    contours_and_sinopsis_flag = contours_flag & sinopsis_flag

    for i, j in itertools.product(range(sinopsis_cube.cube_shape[1]), range(sinopsis_cube.cube_shape[2])):

        if contours_and_sinopsis_flag[i, j] == False:
            print('Skipped')
            continue

        wl_rest = sinopsis_cube.wl / (1 + sinopsis_cube.properties['z'][i, j])

        wl_list.append([wl_rest[0], wl_rest[-1]])
        galaxy_id_list.append(galaxy)

wls = np.array(wl_list)
galaxies = np.array(galaxy_id_list)

galaxies_out_of_range = np.unique(galaxies[wls[:,0] > 3700])
np.savetxt(data_dir + 'no_oii.txt', galaxies_out_of_range, fmt='%s')

print('Common wavelength range from', np.nanmax(wls[:,0]), 'to', np.nanmin(wls[:,1]))

# Picking wavelengths from 3900 to 6000 should work and give some safety wiggle room
# ... but oii is important so we are discarding these: 'A2744_24', 'A2744_25', 'A2744_26', 
# 'A2744_27', 'A2744_28', 'A2744_29'