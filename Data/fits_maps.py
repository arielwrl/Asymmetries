"""

ariel@morelia
14/11/2023

Turns maps stored in pkl into fits files

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import pickle

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Asymetries/Data/'

psb_list = np.genfromtxt(data_dir + 'werle_2020_psb_list.txt', dtype=str).transpose()

center_list = []

for galaxy_id in psb_list:

    print(galaxy_id)

    maps = pickle.load(open(data_dir + 't98.pkl', 'rb'))

    maps[galaxy_id].set_fill_value(-999)

    hdu = fits.PrimaryHDU(maps[galaxy_id])

    hdu_list = fits.HDUList([hdu])
    hdu_list.writeto(data_dir + 'fits_maps_t98/' + galaxy_id + '.fits', overwrite=True)

    pixel_center = np.argwhere(~maps[galaxy_id].mask).mean(axis=0)
    
    center_list.append(pixel_center.tolist())

centers = Table()

centers['galaxy'] = psb_list
centers['x_center'] = np.array(center_list)[:, 1]
centers['y_center'] = np.array(center_list)[:, 0]

centers.write(data_dir + 'fits_maps_massformed/psb_centers.rst', format='ascii.rst',
              overwrite=True)