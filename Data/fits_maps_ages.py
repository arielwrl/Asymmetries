"""

ariel@morelia
16/11/2023

Turns maps stored in pkl into fits files

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from pysinopsis.output import SinopsisCube
import numpy.ma as ma

highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'
data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Asymetries/Data/'
psb_sinopsis_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSBs/SINOPSIS/'
sinopsis_dir = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/'

psb_list = np.genfromtxt(data_dir + 'werle_2020_psb_list.txt', dtype=str).transpose()
others_list = np.genfromtxt(sinopsis_dir + 'file_list.txt', dtype=str).transpose()

center_list = []
bad_list = []

for galaxy_id in others_list:

    print(galaxy_id)

    galaxy_path = sinopsis_dir + galaxy_id + '/'

    try:
        sinopsis_cube = SinopsisCube(galaxy_path)   

        contours = fits.open(highz_dir + 'Data/Contours/' + galaxy_id + '_mask_ellipse.fits')[0].data
        g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy_id + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

    except Exception:
        bad_list.append(galaxy_id)
        continue

    sinopsis_flag = sinopsis_cube.mask == 1
    contours_flag = contours > 2
    contours_and_sinopsis_flag = contours_flag & sinopsis_flag

    mwage_map = ma.masked_array(sinopsis_cube.properties['mwage'], mask=~contours_and_sinopsis_flag)
    # lwage_map = sinopsis_cube.properties['lwage'][contours_and_sinopsis_flag]

    hdu1 = fits.PrimaryHDU(mwage_map)
    # hdu2 = fits.PrimaryHDU(lwage_map)

    hdu_list = fits.HDUList([hdu1])
    hdu_list.writeto(data_dir + 'fits_maps_ages/' + galaxy_id + '.fits', overwrite=True)

    pixel_center = np.argwhere(~contours_and_sinopsis_flag).mean(axis=0)
    
    center_list.append(pixel_center.tolist())

    plt.imshow(mwage_map)
    plt.savefig(data_dir + 'age_maps/' + galaxy_id + '.png')
    plt.close()

centers = Table()

centers['galaxy'] = others_list
centers['x_center'] = np.array(center_list)[:, 1]
centers['y_center'] = np.array(center_list)[:, 0]

centers.write(data_dir + 'fits_maps_ages/centers.rst', format='ascii.rst',
              overwrite=True)

np.savetxt(data_dir + 'bad_list.txt', bad_list.transpose(), fmt=str)

# for galaxy_id in psb_list:

#     hdu = fits.PrimaryHDU(age_map)

#     hdu_list = fits.HDUList([hdu])
#     hdu_list.writeto(data_dir + 'fits_maps_ages/' + galaxy_id + '.fits', overwrite=True)

#     pixel_center = np.argwhere(~age_map.mask).mean(axis=0)
    
#     center_list.append(pixel_center.tolist())
