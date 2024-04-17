"""

ariel@viacordenons
05/01/2024

Generates models for a training set determining the probability of a spectra being quenched

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pysinopsis.output import SinopsisCube
from pysinopsis.utils import calc_manual_ew
import pandas as pd
import pickle
import itertools
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# from toolbox.plot_tools import plot_confusion_matrix
from toolbox.kubevizResults import kubevizResults as kv
import seaborn as sns

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')
galaxy_list = sample['ID']

flag = (sample['Category'] == 'Post-Starburst') | (sample['Category'] == 'Cluster Control') | \
       (sample['Category'] == 'Field Control') | (sample['Category'] == 'Quiescent') | (sample['Category'] == 'Jellyfish')

wl_norm = 5635
common_wl = np.arange(3680, 6001, 3.25)
ew_wl = np.arange(3680, 6001, 3.25)

galaxy_list_models = []
id_list = []
contour_value = []
category_list = []
category_list_original = []
spectrum_list = []
sn_list_5635 = []
sn_list_4050 = []
sn_cont_list = []
ew_dict = {'Hd': [], 'Hb': [], 'Oii': [], 'Oiii': []}

for galaxy in galaxy_list[flag]:

    # if galaxy in ['A2744_07', 'MACS1206_06']:
    #     continue

    galaxy_index = np.argwhere(galaxy_list == galaxy)[0][0]

    if sample['nad_notch'][galaxy_index]:
        continue

    galaxy_path = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/' + galaxy + '/'

    sinopsis_cube = SinopsisCube(galaxy_path)   

    contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
    g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

    sinopsis_flag = sinopsis_cube.mask == 1
    contours_flag = contours >= 1.5
    contours_and_sinopsis_flag = contours_flag & sinopsis_flag

    for i, j in itertools.product(range(sinopsis_cube.cube_shape[1]), range(sinopsis_cube.cube_shape[2])):

        if contours_flag[i, j] == False:
            continue

        if (sample['Category'][galaxy_index] == 'Cluster Control') | (sample['Category'][galaxy_index] == 'Field Control'):
            category_list.append('Star-Forming')
        elif (sample['Category'][galaxy_index] == 'Post-Starburst') | (sample['Category'][galaxy_index] == 'Quiescent'):
            category_list.append('Quenched')
        elif (sample['Category'][galaxy_index] == 'Jellyfish'):
            category_list.append('Jellyfish')
        else:
            print('Dafuq you doing?')
            break

        contour_value.append(contours[i, j])
        id_list.append(galaxy + '_' + str(i) + '_' + str(j))
        category_list_original.append(sample['Category'][galaxy_index])

        wl = sinopsis_cube.wl / (1+sample['z'][galaxy_index])
        flux = sinopsis_cube.f_obs[:, i, j]
        
        flux_norm = flux / np.mean(flux[(wl > wl_norm-50) & (wl < wl_norm+50)])
        
        flux_interp = interp1d(wl, flux_norm)
        flux_template = flux_interp(common_wl)
        flux_template_ew = flux_interp(ew_wl)

        spectrum_list.append(flux_template)
        galaxy_list_models.append(galaxy)

        sn_list_5635.append(np.mean(flux[(wl > 5635-75) & (wl < 5635+75)])/np.std(flux[(wl > 5635-75) & (wl < 5635+75)]))
        sn_list_4050.append(np.mean(flux[(wl > 3995) & (wl < 4075)])/np.std(flux[(wl > 3995) & (wl < 4075)]))

        for line in ew_dict.keys():
            ew_dict[line].append(calc_manual_ew(ew_wl, flux_template_ew, 3.25, line))

categories = np.array(category_list)
spectra = np.array(spectrum_list)
print((categories=='Quenched').sum())
print((categories=='Star-Forming').sum())

models_dict = {}
models_dict['galaxy'] = galaxy_list_models
models_dict['spec_id'] = id_list
models_dict['contours_value'] = contour_value
models_dict['category'] = category_list
models_dict['category_original'] = category_list_original
models_dict['sn_5635'] = sn_list_5635
models_dict['sn_4050'] = sn_list_4050
for line in ew_dict.keys():
    models_dict[line + '_ew'] = ew_dict[line]
models_df = pd.DataFrame(models_dict)

models_df.to_csv(data_dir + 'cnn_models_table_all_3a.csv')

model_spectra_dict = {'model_id': id_list,
                      'category': categories,
                      'spectra': spectra}

pickle.dump(model_spectra_dict, open(data_dir + 'cnn_models_all_3a.pkl', 'wb'))

