"""

ariel@viacordenons
05/01/2024

Generates models to perform a logistic regression determining the probability of a
spectra being post-starburst

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pysinopsis.output import SinopsisCube
from pysinopsis.utils import gini
import pandas as pd
import pickle
import itertools
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from toolbox.plot_tools import plot_confusion_matrix

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Asymetries/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')
galaxy_list = sample['ID']

problematic_cases = np.genfromtxt(data_dir + 'mass_formed_issues.txt', dtype=str)

flag = (sample['Category'] == 'Post=Starburst') | (sample['Category'] == 'Cluster Control') | (sample['Category'] == 'Field Control')

wl_norm = 4020
common_wl = np.arange(3900, 6001, 1)

category_list = []
spectrum_list = []

for galaxy in galaxy_list[flag]:

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

        if (sample['Category'][galaxy_index] == 'Cluster Control') | (sample['Category'][galaxy_index] == 'Field Control'):
            category_list.append('Star-Forming')
        elif sample['Category'][galaxy_index] == 'Post=Starburst':
            category_list.append('Post=Starburst')
        else:
            print('Dafuq you doing?')
            break

        wl = sinopsis_cube.wl / (1+sinopsis_cube.properties['z'][i, j])
        flux = sinopsis_cube.f_obs[:, i, j]
        
        flux_norm = flux / np.mean(flux[(wl > wl_norm-50) & (wl < wl_norm+50)])
        flux_interp = interp1d(wl, flux_norm)
        flux_template = flux_interp(common_wl)

        spectrum_list.append(flux_template)

categories = np.array(category_list)
spectra = np.array(spectrum_list)

training_set = np.arange(0, 0.8*len(spectra), 1).astype(int)
validation_set = np.arange(0.8*len(spectra), len(spectra), 1).astype(int)

classifier = LogisticRegression(random_state=42, max_iter=int(1e4)).fit(spectra[training_set], categories[training_set])
predicted_labels = classifier.predict(spectra[validation_set])
probabilities = classifier.predict_proba(spectra[validation_set])
score = classifier.score(spectra[validation_set], categories[validation_set])

plot_confusion_matrix(confusion_matrix(categories[validation_set], predicted_labels),
                      classes=['Star-Forming', 'Post-Starburst'], normalize=True)