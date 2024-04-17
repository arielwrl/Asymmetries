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
import itertools
from scipy.interpolate import interp1d
import seaborn as sns
from pysinopsis.utils import calc_center_of_mass
import matplotlib.pyplot as plt
from pysinopsis.utils import box_filter
from tensorflow import keras
from pysinopsis.utils import calc_manual_ew

sns.set_style('ticks')

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'
plots_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB regions/Plots/classification_cnn/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')
galaxy_list = sample['ID']

select_list = galaxy_list[(sample['Category'] == 'Jellyfish')].to_list() 

classifier = keras.models.Sequential([
             keras.layers.Conv1D(32, kernel_size=3, padding="same", strides=2, activation="relu", input_shape=(1857, 1)),
             keras.layers.Conv1D(16, kernel_size=3, padding="same", strides=3, activation="relu"),
             keras.layers.MaxPool1D(),
             keras.layers.Flatten(),
             keras.layers.Dropout(0.25),
             keras.layers.Dense(64, activation="relu"),
             keras.layers.Dropout(0.5),
             keras.layers.Dense(2, activation="softmax")
])

classifier.load_weights('Data/cnn_checkpoints/checkpoint_10')

classification_dict = {}

wl_norm = 5635
common_wl = np.arange(3680, 6001, 1.25)
ew_wl = np.arange(3680, 6001, 3.25)

new_columns = {}
new_columns['classification_flag'] = [True if galaxy in select_list else False for galaxy in galaxy_list]
new_columns['n_classified'] = np.empty(len(sample))
new_columns['n_sf'] = np.empty(len(sample))
new_columns['n_psb'] = np.empty(len(sample))
new_columns['n_intermediate'] = np.empty(len(sample))
new_columns['classified_fraction'] = np.empty(len(sample))
new_columns['sf_fraction'] = np.empty(len(sample))
new_columns['psb_fraction'] = np.empty(len(sample))
new_columns['intermediate_fraction'] = np.empty(len(sample))
new_columns['psb_center_x'] = np.empty(len(sample))
new_columns['psb_center_y'] = np.empty(len(sample))

for galaxy in galaxy_list:

    if galaxy not in select_list:
        continue

    galaxy_index = np.argwhere(galaxy_list == galaxy)[0][0]

    galaxy_path = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/' + galaxy + '/'

    sinopsis_cube = SinopsisCube(galaxy_path)   

    contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
    g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

    try:
        ellipse = np.genfromtxt(highz_dir + 'Data/Contours/' + galaxy + '_contours_5s_ellipse.txt').transpose()
    except:
        ellipse = np.genfromtxt(highz_dir + 'Data/Contours/' + galaxy + '_contours_5.0s_ellipse.txt').transpose()

    sinopsis_flag = sinopsis_cube.mask == 1
    
    if galaxy == 'MACS0416S_02':
        contours_flag = contours >= 5
    else:
        contours_flag = contours >= 3
    contours_and_sinopsis_flag = contours_flag & sinopsis_flag

    n_disk = contours_and_sinopsis_flag.sum()

    empty_map = np.ma.masked_array(np.empty_like(sinopsis_cube.properties['mwage']), mask=~contours_and_sinopsis_flag)

    classification_map = np.ma.masked_array(np.empty_like(sinopsis_cube.properties['mwage']).astype(str), mask=~contours_and_sinopsis_flag)
    probability_map = np.ma.masked_array(np.empty_like(sinopsis_cube.properties['mwage']), mask=~contours_and_sinopsis_flag)
    ew_map = {'Hd': np.ma.masked_array(np.empty_like(sinopsis_cube.properties['mwage']), mask=~contours_and_sinopsis_flag), 
              'Hb': np.ma.masked_array(np.empty_like(sinopsis_cube.properties['mwage']), mask=~contours_and_sinopsis_flag),
              'Oii': np.ma.masked_array(np.empty_like(sinopsis_cube.properties['mwage']), mask=~contours_and_sinopsis_flag), 
              'Oiii': np.ma.masked_array(np.empty_like(sinopsis_cube.properties['mwage']), mask=~contours_and_sinopsis_flag)}
    processed_spectra = np.empty(shape=(len(common_wl), sinopsis_cube.cube_shape[1], sinopsis_cube.cube_shape[2]))

    for i, j in itertools.product(range(sinopsis_cube.cube_shape[1]), range(sinopsis_cube.cube_shape[2])):

        if contours_flag[i, j] == False:
            continue

        wl = sinopsis_cube.wl / (1+sample['z'][galaxy_index])
        flux = sinopsis_cube.f_obs[:, i, j]
        
        try:
            flux_norm = flux / np.mean(flux[(wl > wl_norm-50) & (wl < wl_norm+50)])
            flux_interp = interp1d(wl, flux_norm)
            flux_to_classify = flux_interp(common_wl)
            flux_for_ew = flux_interp(ew_wl)
        except:
            probability_map[i, j] = 0
            probability_map.mask[i, j] = True
            continue
        
        if galaxy == 'MACS0940_01':
            sn = np.mean(flux[(wl > 5635-75) & (wl < 5635+75)])/np.std(flux[(wl > 5635-75) & (wl < 5635+75)])
        else:
            sn = np.mean(flux[(wl > 3995) & (wl < 4075)])/np.std(flux[(wl > 3995) & (wl < 4075)])
        
        if sn < 3:
            probability_map[i, j] = 0
            probability_map.mask[i, j] = True
            continue

        processed_spectra[:, i, j] = flux_to_classify

        probability_map[i, j] = classifier.predict(np.array([flux_to_classify]))[0][1]
        classification_map[i, j] = 'Quenched' if probability_map[i, j] > 0.3 else 'Star-Forming' if probability_map[i, j] < 0.3 else 'Intermediate'
        for line in ew_map.keys():
            ew_map[line][i, j] = calc_manual_ew(ew_wl, flux_for_ew, 3.25, line)

    classification_dict[galaxy] = {}
    classification_dict[galaxy]['probability'] = probability_map
    classification_dict[galaxy]['class'] = classification_map
    classification_dict[galaxy]['ew'] = ew_map

    classes_2d = classification_map.ravel()

    n_classified = ((~probability_map.mask) & contours_flag).sum()
    n_psb = (classes_2d == 'Quenched').sum()
    n_sf = (classes_2d == 'Star-Forming').sum()
    n_intermediate = (classes_2d == 'Intermediate').sum()

    y_psb, x_psb = calc_center_of_mass(probability_map, (contours == 5) & ~probability_map.mask)

    new_columns['n_classified'][galaxy_index] = n_classified
    new_columns['n_sf'][galaxy_index] = n_sf
    new_columns['n_psb'][galaxy_index] = n_psb
    new_columns['n_intermediate'][galaxy_index] = n_intermediate
    new_columns['classified_fraction'][galaxy_index] = n_classified / n_disk
    new_columns['sf_fraction'][galaxy_index] = n_sf / n_classified
    new_columns['psb_fraction'][galaxy_index] = n_psb / n_classified
    new_columns['intermediate_fraction'][galaxy_index] = n_intermediate / n_classified
    new_columns['psb_center_x'][galaxy_index] = x_psb
    new_columns['psb_center_y'][galaxy_index] = y_psb

    cmap = sns.color_palette("Spectral_r", as_cmap=True)

    palette = sns.diverging_palette(220, 20, n=10)

    plt.figure()

    plt.title(galaxy)

    plt.imshow(probability_map, origin='lower', cmap=cmap, vmin=0, vmax=1)
    plt.colorbar()

    plt.plot(ellipse[0], ellipse[1], '--k')
    plt.xlim(0.75 * np.min(ellipse[0]), 1.25 * np.max(ellipse[0]))
    plt.ylim(0.75 * np.min(ellipse[1]), 1.25 * np.max(ellipse[1]))

    plt.scatter(x_psb, y_psb, marker='x', color='magenta', s=300, label='Quenching Center')

    plt.savefig(plots_dir + galaxy + '_map.png', dpi=300)

    plt.figure()

    spectra_2d = processed_spectra.reshape((len(common_wl), sinopsis_cube.cube_shape[1] * sinopsis_cube.cube_shape[2])).transpose()
    classes_2d = classification_map.ravel()

    psb_spectra = spectra_2d[classes_2d == 'Quenched']
    sf_spectra = spectra_2d[classes_2d == 'Star-Forming']
    intermediate_spectra = spectra_2d[classes_2d == 'Intermediate']

    wl_psb, flux_psb = box_filter(common_wl, np.median(psb_spectra, axis=0), box_width=5)
    wl_sf, flux_sf = box_filter(common_wl, np.median(sf_spectra, axis=0), box_width=5)
    wl_intermediate, flux_intermediate = box_filter(common_wl, np.median(intermediate_spectra, axis=0), box_width=5)

    plt.plot(wl_intermediate, flux_intermediate, lw=0.5, c='green')
    plt.plot(wl_sf, flux_sf, lw=0.5, c=palette[0])
    plt.plot(wl_psb, flux_psb, lw=0.5, c=palette[-1])
    
    plt.xlim(common_wl[0], common_wl[-1])

    sns.despine()

    plt.savefig(plots_dir + galaxy + '_spectra.png', dpi=300)

    plt.figure()

plt.close('all')

for new_column in new_columns.keys():
    sample[new_column] = new_columns[new_column]

sample['has_quenching'] = sample['sf_fraction'] < 0.1

sample.to_csv(data_dir + 'galaxy_sample.csv', index=False)

pickle.dump(classification_dict, open(data_dir + 'classification_cnn.pkl', 'wb'))

select_list = galaxy_list[(sample['Category'] == 'Jellyfish')]

ew_dict_all = {'galaxy': [], 'class': [], 'Hd': [], 'Hb': [], 'Oii': [], 'Oiii': []}

for galaxy in select_list:

    n_gal = len(classification_dict[galaxy]['class'].ravel())

    for i in range(n_gal):

        if classification_dict[galaxy]['probability'].ravel().mask[i]:
            continue

        ew_dict_all['galaxy'].append(galaxy)
        ew_dict_all['class'].append(classification_dict[galaxy]['class'].ravel()[i])
        ew_dict_all['Hd'].append(classification_dict[galaxy]['ew']['Hd'].ravel()[i])
        ew_dict_all['Hb'].append(classification_dict[galaxy]['ew']['Hb'].ravel()[i])
        ew_dict_all['Oii'].append(classification_dict[galaxy]['ew']['Oii'].ravel()[i])
        ew_dict_all['Oiii'].append(classification_dict[galaxy]['ew']['Oiii'].ravel()[i])

ew_df = pd.DataFrame(ew_dict_all)
ew_df.to_csv(data_dir + 'ew_classes_cnn.csv', index=False)