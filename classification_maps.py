"""

ariel@oapd
16/01/2024

Adds classification to the integrated table, with separate information for different classes

"""

import numpy as np
from astropy.io import fits
from pysinopsis.output import SinopsisCube
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import itertools
from scipy.interpolate import interp1d
import seaborn as sns
from toolbox.kubevizResults import kubevizResults as kv
from pysinopsis.utils import calc_center_of_mass
import matplotlib.pyplot as plt
from pysinopsis.utils import box_filter

sns.set_style('ticks')

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'
plots_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB regions/Plots/classification/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')
properties_table = pd.read_csv(data_dir + 'sample_properties.csv')
regression_models = pickle.load(open(data_dir + 'logistic_regression_models.pkl', 'rb'))
galaxy_list = sample['ID']

select_list = galaxy_list[(sample['Category'] == 'Jellyfish') & (galaxy_list != 'SMACS2131_01') & 
                          (galaxy_list != 'SMACS2131_03') & (galaxy_list != 'MACS0257_02') & (galaxy_list != 'MACS0940_02')
                          & (galaxy_list != 'MACS0416S_02')].to_list() 

categories = np.array(regression_models['category'])
spectra = np.array(regression_models['spectra'])

classifier = LogisticRegression(random_state=42, max_iter=1000).fit(spectra, categories)

classification_dict = {}

wl_norm = 4020
common_wl = np.arange(3680, 6001, 1)

properties_table['classification_flag'] = [True if galaxy in select_list else False for galaxy in galaxy_list]
properties_table['n_sf'] = np.empty(len(properties_table))
properties_table['n_psb'] = np.empty(len(properties_table))
properties_table['n_intermediate'] = np.empty(len(properties_table))
properties_table['sf_fraction'] = np.empty(len(properties_table))
properties_table['psb_fraction'] = np.empty(len(properties_table))
properties_table['intermediate_fraction'] = np.empty(len(properties_table))
properties_table['psb_center_x'] = np.empty(len(properties_table))
properties_table['psb_center_y'] = np.empty(len(properties_table))

for galaxy in galaxy_list:

    if galaxy not in select_list:
        continue

    galaxy_index = np.argwhere(galaxy_list == galaxy)[0][0]

    galaxy_path = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/' + galaxy + '/'

    sinopsis_cube = SinopsisCube(galaxy_path)   

    contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
    g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

    ellipse = np.genfromtxt(highz_dir + 'Data/Masks/' + galaxy + '_contours_5.0s_ellipse.txt').transpose()

    sinopsis_flag = sinopsis_cube.mask == 1
    contours_flag = contours > 3
    contours_and_sinopsis_flag = contours_flag & sinopsis_flag

    n_disk = contours_and_sinopsis_flag.sum()

    classification_map = np.ma.masked_array(np.empty_like(sinopsis_cube.properties['mwage'].astype(str)), mask=~contours_and_sinopsis_flag)
    probability_map = np.ma.masked_array(np.empty_like(sinopsis_cube.properties['mwage']), mask=~contours_and_sinopsis_flag)
    processed_spectra = np.empty(shape=(len(common_wl), sinopsis_cube.cube_shape[1], sinopsis_cube.cube_shape[2]))

    for i, j in itertools.product(range(sinopsis_cube.cube_shape[1]), range(sinopsis_cube.cube_shape[2])):

        if contours_and_sinopsis_flag[i, j] == False:
            continue

        wl = sinopsis_cube.wl / (1+sinopsis_cube.properties['z'][i, j])
        flux = sinopsis_cube.f_obs[:, i, j]
        
        try:
            flux_norm = flux / np.mean(flux[(wl > wl_norm-50) & (wl < wl_norm+50)])
            flux_interp = interp1d(wl, flux_norm)
            flux_to_classify = flux_interp(common_wl)
        except:
            probability_map[i, j] = -1
            continue

        processed_spectra[:, i, j] = flux_to_classify

        probability_map[i, j] = classifier.predict_proba([flux_to_classify])[0][0]
        classification_map[i, j] = 'Post-Starburst' if probability_map[i, j] > 0.8 else 'Star-Forming' if probability_map[i, j] < 0.2 else 'Intermediate'

    classification_dict[galaxy] = {}
    classification_dict[galaxy]['probability'] = probability_map
    classification_dict[galaxy]['class'] = classification_map

    classes_2d = classification_map.ravel()

    n_psb = (classes_2d == 'Post-Starburst').sum()
    n_sf = (classes_2d == 'Star-Forming').sum()
    n_intermediate = (classes_2d == 'Intermediate').sum()

    y_psb, x_psb = calc_center_of_mass(probability_map, contours_and_sinopsis_flag & (probability_map > 0))

    properties_table['n_sf'][galaxy_index] = n_sf
    properties_table['n_psb'][galaxy_index] = n_psb
    properties_table['n_intermediate'][galaxy_index] = n_intermediate
    properties_table['sf_fraction'][galaxy_index] = n_sf / n_disk
    properties_table['psb_fraction'][galaxy_index] = n_psb / n_disk
    properties_table['intermediate_fraction'][galaxy_index] = n_intermediate / n_disk
    properties_table['psb_center_x'][galaxy_index] = x_psb
    properties_table['psb_center_y'][galaxy_index] = y_psb

    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    palette = sns.diverging_palette(220, 20, n=10)

    plt.figure()

    plt.title(galaxy)

    plt.imshow(probability_map, origin='lower', cmap=cmap, vmin=0, vmax=1)
    plt.colorbar()

    plt.plot(ellipse[0], ellipse[1], '--k')
    plt.xlim(0.75 * np.min(ellipse[0]), 1.25 * np.max(ellipse[0]))
    plt.ylim(0.75 * np.min(ellipse[1]), 1.25 * np.max(ellipse[1]))

    plt.scatter(x_psb, y_psb, marker='x', color='magenta', s=300, label='PSB Center')

    plt.savefig(plots_dir + galaxy + '_map.png', dpi=300)

    plt.figure()

    spectra_2d = processed_spectra.reshape((len(common_wl), sinopsis_cube.cube_shape[1] * sinopsis_cube.cube_shape[2])).transpose()
    classes_2d = classification_map.ravel()

    psb_spectra = spectra_2d[classes_2d == 'Post-Starburst']
    sf_spectra = spectra_2d[classes_2d == 'Star-Forming']
    intermediate_spectra = spectra_2d[classes_2d == 'Intermediate']

    wl_psb, flux_psb = box_filter(common_wl, np.median(psb_spectra, axis=0), box_width=10)
    wl_sf, flux_sf = box_filter(common_wl, np.median(sf_spectra, axis=0), box_width=10)
    wl_intermediate, flux_intermediate = box_filter(common_wl, np.median(intermediate_spectra, axis=0), box_width=10)

    plt.plot(wl_intermediate, flux_intermediate, lw=0.5, c='green')
    plt.plot(wl_sf, flux_sf, lw=0.5, c=palette[0])
    plt.plot(wl_psb, flux_psb, lw=0.5, c=palette[-1])
    
    plt.xlim(common_wl[0], common_wl[-1])

    sns.despine()

    plt.savefig(plots_dir + galaxy + '_spectra.png', dpi=300)

    plt.figure()

    sfh_2d = sinopsis_cube.sfh.reshape((sinopsis_cube.sfh.shape[0], sinopsis_cube.sfh.shape[1] * sinopsis_cube.sfh.shape[2])).transpose()
    psb_sfh = sfh_2d[classes_2d == 'Post-Starburst']
    sf_sfh = sfh_2d[classes_2d == 'Star-Forming']
    intermediate_sfh = sfh_2d[classes_2d == 'Intermediate']

    n_psb = (classes_2d == 'Post-Starburst').sum()
    n_sf = (classes_2d == 'Star-Forming').sum()
    n_intermediate = (classes_2d == 'Intermediate').sum()

    median_intermediate_sfh = np.median(intermediate_sfh, axis=0)
    median_sf_sfh = np.median(sf_sfh, axis=0)
    median_psb_sfh = np.median(psb_sfh, axis=0)

    plt.plot(np.log10(sinopsis_cube.age_bin_center)[0:-1], median_intermediate_sfh[0:-1], lw=0.5, c='green', ls='dotted')
    plt.plot(np.log10(sinopsis_cube.age_bin_center)[0:-1], median_sf_sfh[0:-1], lw=0.5, c=palette[0], ls='dotted')
    plt.plot(np.log10(sinopsis_cube.age_bin_center)[0:-1], median_psb_sfh[0:-1], lw=0.5, c=palette[-1], ls='dotted')

    plt.scatter(np.log10(sinopsis_cube.age_bin_center)[0:-1], median_intermediate_sfh[0:-1], lw=0.5, c='green', edgecolors='w', s=50)
    plt.scatter(np.log10(sinopsis_cube.age_bin_center)[0:-1], median_sf_sfh[0:-1], lw=0.5, c=palette[0], edgecolors='w', s=50)
    plt.scatter(np.log10(sinopsis_cube.age_bin_center)[0:-1], median_psb_sfh[0:-1], lw=0.5, c=palette[-1], edgecolors='w', s=50)

    sns.despine()

    plt.savefig(plots_dir + galaxy + '_sfh.png', dpi=300)

plt.close('all')

properties_table['has_psb_regions'] = properties_table['psb_fraction'] > 0.1

properties_table.to_csv(data_dir + 'sample_properties.csv', index=False)

pickle.dump(classification_dict, open(data_dir + 'classification.pkl', 'wb'))