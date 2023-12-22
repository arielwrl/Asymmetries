import numpy as np
import matplotlib.pyplot as plt
from pysinopsis.output import SinopsisCube
import itertools
from toolbox import plot_tools
import pickle
from pysinopsis.utils import calc_mass_formed
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import pandas as pd
from astropy.io import fits

sns.set_style('ticks')

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Asymetries/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')
galaxy_list = sample['ID']

mass_formed_dict = {}

good_list = []
bad_list = []

for galaxy in galaxy_list:

    galaxy_path = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/' + galaxy + '/'

    try:
        sinopsis_cube = SinopsisCube(galaxy_path)   

        contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
        g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

    except Exception:
        bad_list.append(galaxy)
        continue

    good_list.append(galaxy)

    sinopsis_flag = sinopsis_cube.mask == 1
    contours_flag = contours > 2
    contours_and_sinopsis_flag = contours_flag & sinopsis_flag

    sinopsis_cube = SinopsisCube(galaxy_path)

    m05_map = np.full_like(sinopsis_cube.properties['mwage'], np.nan)
    m05_map = np.ma.masked_array(m05_map, mask=~contours_and_sinopsis_flag)

    m1_map = np.full_like(sinopsis_cube.properties['mwage'], np.nan)
    m1_map = np.ma.masked_array(m1_map, mask=~contours_and_sinopsis_flag)

    m15_map = np.full_like(sinopsis_cube.properties['mwage'], np.nan)
    m15_map = np.ma.masked_array(m15_map, mask=~contours_and_sinopsis_flag)

    for i, j in itertools.product(range(sinopsis_cube.cube_shape[1]), range(sinopsis_cube.cube_shape[2])):

        if contours_and_sinopsis_flag[i, j] == False:
            print('Skipped')
            continue

        print('>>>', galaxy, ', Spaxel', i, j)

        m05_map[i, j] = 1-calc_mass_formed(sinopsis_cube.age_bins, sinopsis_cube.sfh[:, i, j],
                                           sinopsis_cube.age_bin_width, 0.5e9)
        m1_map[i, j] = 1-calc_mass_formed(sinopsis_cube.age_bins, sinopsis_cube.sfh[:, i, j],
                                          sinopsis_cube.age_bin_width, 1e9)
        m15_map[i, j] = 1-calc_mass_formed(sinopsis_cube.age_bins, sinopsis_cube.sfh[:, i, j],
                                           sinopsis_cube.age_bin_width, 1.5e9)

    mass_formed_dict[galaxy] = {}
    mass_formed_dict[galaxy]['m05'] = 100 * m05_map
    mass_formed_dict[galaxy]['m1'] = 100 * m1_map
    mass_formed_dict[galaxy]['m15'] = 100 * m15_map

    mass_formed_list = ['m05', 'm1', 'm15']
    mass_formed_labels = [r'$t < 500\,\mathrm{Myr}$', r'$t < 1\,\mathrm{Gyr}$', r'$t < 1.5\,\mathrm{Gyr}$']

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    for i in range(3):
        flag = ~mass_formed_dict[galaxy][mass_formed_list[i]].mask
        min_mass = np.percentile(mass_formed_dict[galaxy][mass_formed_list[i]][flag], 20)
        max_mass = np.percentile(mass_formed_dict[galaxy][mass_formed_list[i]][flag], 80)

        mass_map = ax[i].imshow(mass_formed_dict[galaxy][mass_formed_list[i]], origin='lower', cmap='plasma',
                                vmin=min_mass, vmax=max_mass)

        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')

        cbaxes = inset_axes(ax[i], width="50%", height="5%", loc='upper left')
        cb = plt.colorbar(cax=cbaxes, label=mass_formed_labels[i], orientation='horizontal', mappable=mass_map)

    plt.savefig('C:/Users/ariel/Workspace/GASP/High-z/PSB Asymetries/Plots/masses_formed/' + galaxy + '.png', dpi=300)
    plt.close()

np.savetxt(data_dir + 'mass_formed_issues.txt', bad_list, fmt='%s')

pickle.dump(mass_formed_dict, open(data_dir + 'mass_formed.pkl', 'wb'))