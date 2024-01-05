import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pysinopsis.output import SinopsisCube
from pysinopsis.utils import gini
import pandas as pd
import pickle
import itertools

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Asymetries/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')
galaxy_list = sample['ID']

problematic_cases = np.genfromtxt(data_dir + 'mass_formed_issues.txt', dtype=str)

wl_list = []

for galaxy in galaxy_list:

    galaxy_index = np.argwhere(galaxy_list == galaxy)[0][0]

    if galaxy in problematic_cases:
        continue

    galaxy_path = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/' + galaxy + '/'

    sinopsis_cube = SinopsisCube(galaxy_path)   

    contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
    g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

    sinopsis_flag = sinopsis_cube.mask == 1
    contours_flag = contours > 2
    contours_and_sinopsis_flag = contours_flag & sinopsis_flag

    for i, j in itertools.product(range(sinopsis_cube.cube_shape[1]), range(sinopsis_cube.cube_shape[2])):

        if contours_and_sinopsis_flag[i, j] == False:
            print('Skipped')
            continue

        wl_rest = sinopsis_cube.wl / (1+sinopsis_cube.properties['z'][i, j])

        wl_list.append([wl_rest[0], wl_rest[-1]])

wls = np.array(wl_list)

print('Common wavelength range from', np.nanmax(wls[:,0]), 'to', np.nanmin(wls[:,1]))


# Picking wavelengths from 3900 to 6000 should work and give some safety wiggle room