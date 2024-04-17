"""

ariel@oapd
18/01/2024

Uses clustering algorithms to isolate the tail of jellyfish galaxies from noise

"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from pysinopsis.output import SinopsisCube
from pysinopsis.utils import calc_center_of_mass
import pandas as pd
from astropy.wcs import WCS
import seaborn as sns
from toolbox.kubevizResults import kubevizResults as kv
from skimage.measure import regionprops

sns.set_style('ticks')

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'
plots_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB regions/Plots/maps_and_centers/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')

galaxy_list = sample['ID']

select_list = galaxy_list[(sample['Category'] == 'Jellyfish')].to_list() 

new_columns = {}
new_columns['has_emission_cube'] = np.full(len(sample), True)
new_columns['oii_center_x'] = np.empty(len(sample))
new_columns['oii_center_y'] = np.empty(len(sample))
new_columns['oii_out_center_x'] = np.empty(len(sample))
new_columns['oii_out_center_y'] = np.empty(len(sample))
new_columns['g_center_x'] = np.empty(len(sample))
new_columns['g_center_y'] = np.empty(len(sample))
new_columns['dot_product'] = np.empty(len(sample))
new_columns['dot_product_out'] = np.empty(len(sample))
new_columns['oii_displacement'] = np.empty(len(sample))
new_columns['oii_displacement_out'] = np.empty(len(sample))
new_columns['dot_product_normed'] = np.empty(len(sample))
new_columns['dot_product_out_normed'] = np.empty(len(sample))
new_columns['oii_displacement_normed'] = np.empty(len(sample))
new_columns['oii_displacement_out_normed'] = np.empty(len(sample))
new_columns['circularized_radius'] = np.empty(len(sample))

for galaxy in galaxy_list:

      if galaxy not in select_list:
            continue

      galaxy_index = np.argwhere(galaxy_list == galaxy)[0][0]

      galaxy_path = highz_dir + 'SINOPSIS/' + galaxy + '/'

      sinopsis_cube = SinopsisCube(galaxy_path)   

      try:
            ellipse = np.genfromtxt(highz_dir + 'Data/Contours/' + galaxy + '_contours_5s_ellipse.txt').transpose()
      except:
            ellipse = np.genfromtxt(highz_dir + 'Data/Contours/' + galaxy + '_contours_5.0s_ellipse.txt').transpose()

      contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
      g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

      try:
            emission_only = kv(highz_dir + 'Data/Emission/' + galaxy + '_DATACUBE_FINAL_v1_ec_eo_res_gau_spax_noflag.fits')

            oii_raw = emission_only.get_flux('o2t')
            sn_oii = emission_only.get_snr('o2t')
            sn_mask = (oii_raw <= 0) | (sn_oii < 3) | np.isnan(oii_raw) | np.isnan(sn_oii)| np.isinf(oii_raw) | np.isinf(sn_oii)
            oii = np.ma.masked_array(oii_raw, mask=sn_mask)
            oii_label_image = (~sn_mask)
            oii[np.isnan(oii) | np.isinf(oii)] = 0

      except:
            new_columns['has_emission_cube'][galaxy_index] = False
            continue

      if galaxy == 'A2744_14':
            new_columns['has_emission_cube'][galaxy_index] = False
            continue

      sinopsis_flag = sinopsis_cube.mask == 1
      contours_flag = contours > 3
      contours_and_sinopsis_flag = contours_flag & sinopsis_flag

      g_center_y, g_center_x = calc_center_of_mass(g_band, contours >= 5)

      oii_center_y, oii_center_x = calc_center_of_mass(oii, ~oii.mask)

      oii_center_out_of_disk_y, oii_center_out_of_disk_x = calc_center_of_mass(oii, (~oii.mask) & (~contours_flag))

      galaxy_properties = sample[sample['ID'] == galaxy]
      psb_center_x, psb_center_y = galaxy_properties['psb_center_x'].array[0], galaxy_properties['psb_center_y'].array[0]

      tail_vector = (oii_center_x - g_center_x, oii_center_y - g_center_y)
      tail_vector_out = (oii_center_out_of_disk_x - g_center_x, oii_center_out_of_disk_y - g_center_y)
      psb_vector = (psb_center_x - g_center_x, psb_center_y - g_center_y)

      dot_product = np.dot(psb_vector, tail_vector / np.linalg.norm(tail_vector))
      dot_product_out = np.dot(psb_vector, tail_vector_out / np.linalg.norm(tail_vector_out))

      region_properties = regionprops(contours_and_sinopsis_flag.astype(int))[0]
      center = region_properties.centroid
      a, b = region_properties.major_axis_length/2, region_properties.minor_axis_length/2
      circularized_radius = np.mean([a, b])

      new_columns['circularized_radius'][galaxy_index] = circularized_radius
      new_columns['oii_center_x'][galaxy_index] = oii_center_x
      new_columns['oii_center_y'][galaxy_index] = oii_center_y
      new_columns['oii_out_center_x'][galaxy_index] = oii_center_out_of_disk_x
      new_columns['oii_out_center_y'][galaxy_index] = oii_center_out_of_disk_y
      new_columns['g_center_x'][galaxy_index] = g_center_x
      new_columns['g_center_y'][galaxy_index] = g_center_y
      new_columns['dot_product'][galaxy_index] = dot_product
      new_columns['dot_product_out'][galaxy_index] = dot_product_out
      new_columns['dot_product_normed'][galaxy_index] = dot_product / circularized_radius
      new_columns['dot_product_out_normed'][galaxy_index] = dot_product_out / circularized_radius
      new_columns['oii_displacement'][galaxy_index] = np.linalg.norm(tail_vector)
      new_columns['oii_displacement_normed'][galaxy_index] = np.linalg.norm(tail_vector) / circularized_radius
      new_columns['oii_displacement_out'][galaxy_index] = np.linalg.norm(tail_vector_out)
      new_columns['oii_displacement_out_normed'][galaxy_index] = np.linalg.norm(tail_vector_out) / circularized_radius

      plt.figure(figsize=(8, 8))

      wcs = WCS(sinopsis_cube.obs_header['flux']).celestial

      ax = plt.subplot(1, 1, 1, projection=wcs)

      plt.imshow(oii, origin='lower', alpha=0.5)
      plt.scatter(g_center_x, g_center_y, marker='x', color='cyan', s=300, label='G-band Center')
      plt.scatter(oii_center_out_of_disk_x, oii_center_out_of_disk_y, marker='x', color='forestgreen', s=300, label='OII Center (out)')
      plt.scatter(galaxy_properties['psb_center_x'], galaxy_properties['psb_center_y'], marker='x', color='indigo', s=300, label='Quenching Center')

      plt.legend()

      vectors = np.array([psb_vector, tail_vector_out])
      origin = (g_center_x, g_center_y)
      for i in range(2):
            plt.quiver(origin[0], origin[1], vectors[i, 0], vectors[i, 1], angles='xy', 
                       scale_units='xy', scale=1, color='k')
      
      circle = plt.Circle(origin, circularized_radius, fill=False, color='k', ls='dashed')
      plt.gca().add_patch(circle)

      plt.xlabel(' ')
      plt.ylabel(' ')

      plt.savefig(plots_dir + galaxy + '.png', dpi=300)

for new_column in new_columns.keys():
    sample[new_column] = new_columns[new_column]

sample.to_csv(data_dir + 'galaxy_sample.csv', index=False)
