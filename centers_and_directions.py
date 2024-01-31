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
import itertools
from scipy.ndimage import gaussian_filter
import seaborn as sns
from toolbox.kubevizResults import kubevizResults as kv
from skimage.measure import regionprops

sns.set_style('ticks')

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'
plots_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB regions/Plots/maps_and_centers/'

sample = pd.read_csv(data_dir + 'galaxy_sample.csv')
properties_table = pd.read_csv(data_dir + 'sample_properties.csv')

galaxy_list = sample['ID']

select_list = galaxy_list[(sample['Category'] == 'Jellyfish') & (galaxy_list != 'SMACS2131_01') & 
                          (galaxy_list != 'SMACS2131_03') & (galaxy_list != 'MACS0257_02') & 
                          (galaxy_list != 'MACS0940_02') & (galaxy_list != 'MACS0416S_02')].to_list() 

properties_table['has_emission'] = np.full(len(properties_table), True)
properties_table['oii_center_x'] = np.empty(len(properties_table))
properties_table['oii_center_y'] = np.empty(len(properties_table))
properties_table['oii_out_center_x'] = np.empty(len(properties_table))
properties_table['oii_out_center_y'] = np.empty(len(properties_table))
properties_table['mass_center_x'] = np.empty(len(properties_table))
properties_table['mass_center_y'] = np.empty(len(properties_table))
properties_table['dot_product'] = np.empty(len(properties_table))
properties_table['dot_product_out'] = np.empty(len(properties_table))
properties_table['oii_displacement'] = np.empty(len(properties_table))
properties_table['oii_displacement_out'] = np.empty(len(properties_table))
properties_table['dot_product_normed'] = np.empty(len(properties_table))
properties_table['dot_product_out_normed'] = np.empty(len(properties_table))
properties_table['oii_displacement_normed'] = np.empty(len(properties_table))
properties_table['oii_displacement_out_normed'] = np.empty(len(properties_table))

for galaxy in galaxy_list:

      # galaxy = 'A2744_06'

      if galaxy not in select_list:
            continue

      galaxy_index = np.argwhere(galaxy_list == galaxy)[0][0]

      galaxy_path = highz_dir + 'SINOPSIS/' + galaxy + '/'

      sinopsis_cube = SinopsisCube(galaxy_path)   

      contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
      g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

      ellipse = np.genfromtxt(highz_dir + 'Data/Masks/' + galaxy + '_contours_5.0s_ellipse.txt').transpose()

      try:
            emission_only = kv(highz_dir + 'Data/Emission/' + galaxy + '_DATACUBE_FINAL_v1_ec_eo_res_gau_spax_noflag.fits')
      except:
            properties_table['has_emission'][galaxy_index] = False
            continue

      if galaxy == 'A2744_14':
            # THIS GALAXY DOESNT HAVE OII
            properties_table['has_emission'][galaxy_index] = False
            continue

      oii_raw = emission_only.get_flux('o2t')
      sn_oii = emission_only.get_snr('o2t')
      sn_mask = (oii_raw <= 0) | (sn_oii < 3) | np.isnan(oii_raw) | np.isnan(sn_oii)| np.isinf(oii_raw) | np.isinf(sn_oii)
      oii = np.ma.masked_array(oii_raw, mask=sn_mask)
      oii_label_image = (~sn_mask)
      oii[np.isnan(oii) | np.isinf(oii)] = 0

      sinopsis_flag = sinopsis_cube.mask == 1
      contours_flag = contours > 3
      contours_and_sinopsis_flag = contours_flag & sinopsis_flag

      mass_center_y, mass_center_x = sinopsis_cube.get_center_of_mass('TotMass2', custom_mask=contours_and_sinopsis_flag)

      oii_center_y, oii_center_x = calc_center_of_mass(oii, ~oii.mask)

      oii_center_out_of_disk_y, oii_center_out_of_disk_x = calc_center_of_mass(oii, (~oii.mask) & (~contours_flag))

      galaxy_properties = properties_table[properties_table['galaxy'] == galaxy]
      psb_center_x, psb_center_y = galaxy_properties['psb_center_x'].array[0], galaxy_properties['psb_center_y'].array[0]

      tail_vector = (oii_center_x - mass_center_x, oii_center_y - mass_center_y)
      tail_vector_out = (oii_center_out_of_disk_x - mass_center_x, oii_center_out_of_disk_y - mass_center_y)
      psb_vector = (psb_center_x - mass_center_x, psb_center_y - mass_center_y)

      dot_product = np.dot(psb_vector, tail_vector / np.linalg.norm(tail_vector))
      dot_product_out = np.dot(psb_vector, tail_vector_out / np.linalg.norm(tail_vector_out))

      region_properties = regionprops(contours_and_sinopsis_flag.astype(int))[0]
      center = region_properties.centroid
      a, b = region_properties.major_axis_length/2, region_properties.minor_axis_length/2
      circularized_radius = np.mean([a, b])
      print(circularized_radius)

      properties_table['oii_center_x'][galaxy_index] = oii_center_x
      properties_table['oii_center_y'][galaxy_index] = oii_center_y
      properties_table['oii_out_center_x'][galaxy_index] = oii_center_out_of_disk_x
      properties_table['oii_out_center_y'][galaxy_index] = oii_center_out_of_disk_y
      properties_table['mass_center_x'][galaxy_index] = mass_center_x
      properties_table['mass_center_y'][galaxy_index] = mass_center_y
      properties_table['dot_product'][galaxy_index] = dot_product
      properties_table['dot_product_out'][galaxy_index] = dot_product_out
      properties_table['dot_product_normed'][galaxy_index] = dot_product / circularized_radius
      properties_table['dot_product_out_normed'][galaxy_index] = dot_product_out / circularized_radius
      properties_table['oii_displacement'][galaxy_index] = np.linalg.norm(tail_vector)
      properties_table['oii_displacement_normed'][galaxy_index] = np.linalg.norm(tail_vector) / circularized_radius
      properties_table['oii_displacement_out'][galaxy_index] = np.linalg.norm(tail_vector_out)
      properties_table['oii_displacement_out_normed'][galaxy_index] = np.linalg.norm(tail_vector_out) / circularized_radius

      plt.figure(figsize=(8, 8))

      plt.title(galaxy + ', dot=' + str(dot_product / circularized_radius) + ', dot-out=' + str(dot_product_out / circularized_radius))
      plt.imshow(oii, origin='lower', alpha=0.5)
      plt.contour(np.ma.masked_array(sinopsis_cube.properties['TotMass2'], mask=~contours_and_sinopsis_flag), 
                  cmap='Greys', zorder=1)
      plt.scatter(mass_center_x, mass_center_y, marker='x', color='red', s=300, label='Mass Center')
      plt.scatter(oii_center_x, oii_center_y, marker='x', color='green', s=300, label='OII Center')
      plt.scatter(oii_center_out_of_disk_x, oii_center_out_of_disk_y, marker='x', color='cyan', s=300, label='OII Center (out)')
      plt.scatter(galaxy_properties['psb_center_x'], galaxy_properties['psb_center_y'], marker='x', color='magenta', s=300, label='PSB Center')

      plt.legend()

      # plt.figure(figsize=(8, 8))

      vectors = np.array([psb_vector, tail_vector, tail_vector_out])
      origin = (mass_center_x, mass_center_y)
      for i in range(3):
            plt.quiver(origin[0], origin[1], vectors[i, 0], vectors[i, 1], angles='xy', 
                       scale_units='xy', scale=1, color='k')
      
      circle = plt.Circle(origin, circularized_radius, fill=False, color='k', ls='dashed')
      plt.gca().add_patch(circle)

      plt.savefig(plots_dir + galaxy + '.png', dpi=300)


plt.figure()

flag = properties_table['classification_flag'] & (properties_table['sf_fraction'] < 0.95) & (properties_table['galaxy'] != 'A2744_01')

plt.scatter(properties_table['oii_displacement_out_normed'][flag], 
            properties_table['dot_product_normed'][flag], s=2500 * properties_table['psb_fraction'][flag],  
            color='firebrick', edgecolors='w')
plt.ylim(-1, 0.17)

mostly_sf_flag = properties_table['classification_flag'] & (properties_table['sf_fraction'] > 0.95) & properties_table['has_emission']

ax = plt.gca()
inset_ax = ax.inset_axes([0, 0.0, 1, 0.2])
inset_ax.set_xticklabels([])
inset_ax.set_yticklabels([])
inset_ax.spines["top"].set_visible(False)
inset_ax.spines["right"].set_visible(False)
inset_ax.patch.set_alpha(0)
inset_ax.tick_params(axis='both', length=0)

sns.boxplot(x=properties_table['oii_displacement_out_normed'][mostly_sf_flag].tolist(), 
            ax=inset_ax,  whis=[0, 90], color='steelblue',
            flierprops = dict(marker='o', markerfacecolor='none', markersize=10,
                  linestyle='none', markeredgecolor='steelblue'))

inset_ax.set_xlim(0, 4.6)
plt.xlim(0, 4.6)

ax.axhline(y=0, linestyle='solid', color='k', linewidth=0.5)

sns.despine()

plt.xlabel(r'$[\mathrm{\sc{OII}}]\lambda3727$ Displacement', fontsize=15)
plt.ylabel('Quenching displacement', fontsize=15)

plt.savefig('C:/Users/ariel/Workspace/GASP/High-z/PSB regions/Plots/diplacements.png', dpi=300)


plt.figure()

classification_flag = properties_table['classification_flag'] & properties_table['has_emission']

# plt.hist(properties_table['intermediate_fraction'][classification_flag], range=[0, 1], bins=10, color='forestgreen', 
#          histtype='step')
plt.hist(properties_table['sf_fraction'][classification_flag], range=[0, 1], bins=10, color='steelblue', 
         histtype='step', label='Star-Forming')
plt.hist(properties_table['psb_fraction'][classification_flag], range=[0, 1], bins=10, color='firebrick', 
         histtype='step', label='Post-Starburst')

plt.xlabel('Fraction Of Spaxels', fontsize=15)
plt.ylabel('Number of Galaxies', fontsize=15)

plt.legend(frameon=False, fontsize=15)

sns.despine()

plt.savefig('C:/Users/ariel/Workspace/GASP/High-z/PSB regions/Plots/fractions.png', dpi=300)