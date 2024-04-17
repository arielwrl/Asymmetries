"""

ariel@oapd
22/01/2024

Uses clustering algorithms to isolate the tail of jellyfish galaxies from noise

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pysinopsis.output import SinopsisCube
from pysinopsis.utils import calc_center_of_mass
import pandas as pd
import itertools
from scipy.ndimage import gaussian_filter
import seaborn as sns
from toolbox.kubevizResults import kubevizResults as kv
from sklearn.cluster import DBSCAN
from skimage.measure import regionprops

sns.set_style('ticks')

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'
plots_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB regions/Plots/dbscan/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')
properties_table = pd.read_csv(data_dir + 'sample_properties.csv')

galaxy_list = sample['ID']

select_list = galaxy_list[sample['Category'] == 'Jellyfish'].to_list() 

galaxy = 'A370_06'

galaxy_index = np.argwhere(galaxy_list == galaxy)[0][0]

galaxy_path = highz_dir + 'SINOPSIS/' + galaxy + '/'

sinopsis_cube = SinopsisCube(galaxy_path)   

contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

ellipse = np.genfromtxt(highz_dir + 'Data/Contours/' + galaxy + '_contours_5s_ellipse.txt').transpose()

emission_only = kv(highz_dir + 'Data/Emission/' + galaxy + '_DATACUBE_FINAL_v1_ec_eo_res_gau_spax_noflag.fits')

oii_raw = emission_only.get_flux('o2t')
sn_oii = emission_only.get_snr('o2t')
sn_mask = (oii_raw <= 0) | (sn_oii < 3) | np.isnan(oii_raw) | np.isnan(sn_oii)| np.isinf(oii_raw) | np.isinf(sn_oii)
oii = np.ma.masked_array(oii_raw, mask=sn_mask)
oii_label_image = (~sn_mask)
# oii[np.isnan(oii)] = 0    

sinopsis_flag = sinopsis_cube.mask == 1
contours_flag = contours > 3
contours_and_sinopsis_flag = contours_flag & sinopsis_flag

coordinates = np.empty(shape=(2, sinopsis_cube.sfh.shape[2] * sinopsis_cube.sfh.shape[1])).transpose()

for i, j in itertools.product(range(sinopsis_cube.cube_shape[2]), range(sinopsis_cube.cube_shape[1])):
    coordinates[j * sinopsis_cube.cube_shape[2] + i] = [i, j]

oii_flag = oii_label_image.ravel()
oii_sn_1d = sn_oii.ravel()

oii_image_coordinates = coordinates[oii_flag]

clustering = DBSCAN(eps=2, min_samples=8).fit(oii_image_coordinates)

# plt.imshow(oii_label_image, origin='lower', cmap='Greys', alpha=0.5)
# plt.scatter(oii_image_coordinates[:,0][clustering.labels_>=0], oii_image_coordinates[:,1][clustering.labels_>=0], 
#             s=3, c=clustering.labels_[clustering.labels_>=0], cmap='Set3')
# plt.colorbar()
# plt.show()

cluster_size = np.array([(clustering.labels_ == label).sum() for label in clustering.labels_])
cluster_labels = np.unique(clustering.labels_)
n_clusters = len(cluster_labels)
largest_cluster_size = np.max(cluster_size)

sn_cluster = np.array([np.mean(oii_sn_1d[oii_flag][(clustering.labels_ == clustering.labels_[i])]) 
                       for i in range(oii_flag.sum())])

oii_velocity = emission_only.data['narrowlineo2tdv']
oii_velocity_1d = oii_velocity.ravel()
velocity_cluster = np.array([np.mean(oii_velocity_1d[oii_flag][(clustering.labels_ == clustering.labels_[i])]) 
                       for i in range(oii_flag.sum())])

plt.figure()
plt.imshow(oii_label_image, origin='lower', cmap='Greys', alpha=0.5)
plt.scatter(oii_image_coordinates[:,0][clustering.labels_>=0], oii_image_coordinates[:,1][clustering.labels_>=0], 
            s=40, c=sn_cluster[clustering.labels_>=0], cmap=sns.color_palette("crest", as_cmap=True), 
            marker='s')
cb = plt.colorbar()
cb.set_label(r'Cluster S/N', size=20)

plt.savefig(plots_dir + galaxy + '_sn.png', dpi=300)
plt.show()


plt.figure()
plt.imshow(oii_label_image, origin='lower', cmap='Greys', alpha=0.5)
plt.scatter(oii_image_coordinates[:,0][clustering.labels_>=0], oii_image_coordinates[:,1][clustering.labels_>=0], 
            s=40, c=oii_velocity_1d[oii_flag][clustering.labels_>=0], cmap='Spectral_r', marker='s')
cb = plt.colorbar()
cb.set_label('$v_{\mathrm{\sc{OII}}} \; \mathrm{[km/s]}$', size=20)

plt.savefig(plots_dir + galaxy + '_vel.png', dpi=300)
plt.show()