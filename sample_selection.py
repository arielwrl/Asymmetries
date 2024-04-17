import numpy as np
import pandas as pd
from pysinopsis.output import SinopsisCube
from astropy.io import fits
import itertools
from toolbox.wololo import ergstoabmags, redshift2lumdistance, arcsectokpc
from toolbox.kubevizResults import kubevizResults as kv

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Data/'
highz_dir = 'C:/Users/ariel/Workspace/GASP/High-z/'
plot_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Regions/Plots/'
emission_dir = 'C:/Users/ariel/Workspace/GASP/High-z/Data/Emission/'

sample = pd.read_table(data_dir + 'sample_v2.dat', delim_whitespace=True)

ellipse_parameters = pd.read_table(highz_dir + 'Data/ellipse_axes_v2.cat', delim_whitespace=True)
ellipse_parameters = ellipse_parameters.sort_values('name')

blended_list = ['A2744_05', 'A2744_12', 'A2744_14', 'A2744_29', 'A370_20', 'A370_24', 'AS1063SW_01', 
                'MACS0416S_01', 'MACS0940_02', 'MACS0940_05', 'MACS1206_04', 'SMACS2031_03', 
                'SMACS2131_06']

sample['RPS_class_code'] = sample['RPS_class'].copy()

sample['Dec'] = sample['DEC']
sample['Ha_in'] = sample['Ha_in'].astype(bool)

sample['RPS_class'][sample['RPS_class_code'] == 0] = 'Unaffected'
sample['RPS_class'][sample['RPS_class_code'] == 1] = 'Jellyfish'
sample['RPS_class'][sample['RPS_class_code'] == 2] = 'Post-Starburst'
sample['RPS_class'][sample['RPS_class_code'] == 3] = 'Truncated Disk'
sample['RPS_class'][sample['RPS_class_code'] < 0] = 'Unclassified'

sample['RPS_class'][sample['ID'] == 'AS1063NE_07'] = 'Unclassified'
sample['RPS_class'][sample['ID'] == 'MACS1206_01'] = 'Merger/Outflow'

sample['Category'] = np.full_like(sample['RPS_class'], fill_value='Pending')
sample['Category'][sample['RPS_class'] == 'Post-Starburst'] = 'Post-Starburst'
sample['Category'][sample['RPS_class'] == 'Truncated Disk'] = 'Truncated Disk'
sample['Category'][sample['RPS_class'] == 'Jellyfish'] = 'Jellyfish'
sample['Category'][(sample['RPS_class'] == 'Unaffected') & sample['Memb'].astype(bool) & 
                sample['gas_disk'].astype(bool)] = 'Cluster Control'
sample['Category'][(sample['RPS_class'] == 'Unaffected') & ~sample['Memb'].astype(bool) & 
                sample['gas_disk'].astype(bool)] = 'Field Control'
sample['Category'][(sample['gas_disk'] == 0) & (sample['RPS_class_code'] == 0) & sample['Memb'].astype(bool)] = 'Quiescent'

sample_flag = np.array([category in ['Post-Starburst', 'Truncated Disk', 'Jellyfish', 
                                      'Cluster Control', 'Field Control', 'Quiescent'] for 
                                      category in sample['Category']])

final_sample = sample[['ID', 'RA', 'Dec', 'z', 'Category', 'Ha_in']].set_index('ID')
final_sample['Category'][final_sample.index == 'A2744_01'] = 'Post-Starburst'

bad_contours_list = []
bad_gband_list = []
bad_sinopsis_list = []
good_list = []

final_sample['n_spaxels'] = np.empty(shape=len(final_sample))
final_sample['n_disk'] = np.empty(shape=len(final_sample))
final_sample['n_spaxels_disk'] = np.empty(shape=len(final_sample))
final_sample['flux_spaxels'] = np.empty(shape=len(final_sample))
final_sample['flux_disk'] = np.empty(shape=len(final_sample))
final_sample['flux_spaxels_disk'] = np.empty(shape=len(final_sample))
final_sample['sn_5635'] = np.empty(shape=len(final_sample))
final_sample['sn_4050'] = np.empty(shape=len(final_sample))
final_sample['bad_sinopsis'] = np.full(fill_value=False, shape=len(final_sample))
final_sample['bad_contours'] = np.full(fill_value=False, shape=len(final_sample))
final_sample['bad_gband'] = np.full(fill_value=False, shape=len(final_sample))
final_sample['nad_notch'] = np.full(fill_value=False, shape=len(final_sample))
final_sample['blended'] = np.full(fill_value=False, shape=len(final_sample))

wl_norm = 5635

galaxy_list = final_sample.index

for galaxy in galaxy_list:

    galaxy_path = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/' + galaxy + '/'
    galaxy_index = np.argwhere(final_sample.index == galaxy)[0][0]

    if galaxy in blended_list:
        final_sample['blended'][galaxy_index] = True

    if (galaxy.split('_')[0] in ['MACS0257', 'RXJ1347', 'SMACS2131']) & (galaxy not in ['RXJ1347_03', 'RXJ1347_07']):
        final_sample['nad_notch'][galaxy_index] = True

    try:
        sinopsis_cube = SinopsisCube(galaxy_path)   
    except:
        bad_sinopsis_list.append(galaxy)
        final_sample['bad_sinopsis'][galaxy_index] = True
        continue

    try:
        contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
    except:
        bad_contours_list.append(galaxy)
        final_sample['bad_contours'][galaxy_index] = True
        continue

    try:
        g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data
    except:
        bad_gband_list.append(galaxy)
        final_sample['bad_gband'][galaxy_index] = True
        continue

    good_list.append(galaxy)

    sinopsis_flag = sinopsis_cube.mask == 1
    contours_flag = contours >= 5
    contours_and_sinopsis_flag = contours_flag & sinopsis_flag

    sn_sinopsis = []
    spec_list = []
    spec_list_5635 = []
    spec_list_4050 = []

    for i, j in itertools.product(range(sinopsis_cube.cube_shape[1]), range(sinopsis_cube.cube_shape[2])):

        if contours_flag[i, j] == False:
            print('Skipped')
            continue

        wl = sinopsis_cube.wl / (1 + final_sample['z'][galaxy_index])
        flux = sinopsis_cube.f_obs[:, i, j]
        errors = sinopsis_cube.f_err[:, i, j]

        flag = (flux > 0) & (errors > 0)

        spec_list_5635.append(flux[(wl > wl_norm-75) & (wl < wl_norm+75)])
        spec_list_4050.append(flux[(wl > 3995) & (wl < 4075)])
        spec_list.append(flux)

    spec_list = np.array(spec_list)
    spec_list_5635 = np.array(spec_list_5635)
    spec_list_4050 = np.array(spec_list_4050)
    summed_spectrum = spec_list.sum(axis=0)
    summed_spectrum_5635 = spec_list_5635.sum(axis=0)
    summed_spectrum_4050 = spec_list_4050.sum(axis=0)

    n_spaxels = sinopsis_flag.sum()
    n_disk = contours_flag.sum()
    n_spaxels_disk = contours_and_sinopsis_flag.sum()
    flux_spaxels = np.ma.masked_array(g_band, mask=~sinopsis_flag).sum()
    flux_disk = np.ma.masked_array(g_band, mask=~contours_flag).sum()
    flux_spaxels_disk = np.ma.masked_array(g_band, mask=~contours_and_sinopsis_flag).sum()

    final_sample['n_spaxels'][galaxy_index] = n_spaxels
    final_sample['n_disk'][galaxy_index] = n_disk
    final_sample['n_spaxels_disk'][galaxy_index] = n_spaxels_disk
    final_sample['flux_spaxels'][galaxy_index] = flux_spaxels
    final_sample['flux_disk'][galaxy_index] = flux_disk
    final_sample['flux_spaxels_disk'][galaxy_index] = flux_spaxels_disk
    final_sample['sn_5635'][galaxy_index] = np.mean(summed_spectrum_5635)/np.std(summed_spectrum_5635)
    final_sample['sn_4050'][galaxy_index] = np.mean(summed_spectrum_4050)/np.std(summed_spectrum_4050)

final_sample['mag_disk'] = ergstoabmags(final_sample['flux_disk'] * 1e-20, wl=4770)
final_sample['sinopsis_fraction'] = final_sample['n_spaxels_disk']/final_sample['n_disk']
final_sample['sinopsis_fraction_flag'] = final_sample['sinopsis_fraction'] > 0.5

quality_flag = (final_sample['mag_disk'] < 25)  & (final_sample['z'] > 0.27) & sample_flag & (~final_sample['blended']) & (~final_sample['nad_notch'])
final_sample[~quality_flag].to_csv(data_dir + 'rejected_sample.csv')
final_sample.to_csv(data_dir + 'all_galaxy_sample.csv')
final_sample = final_sample[quality_flag]

# SMACS2131_01 in the edge of the field, A2744_14 has inconclusive redshift and might not be member

# Compile stellar masses and SFRs (formely compile_data.py):

galaxy_list = final_sample.index

new_columns = {}
new_columns['mass'] = np.empty(shape=len(galaxy_list))
new_columns['mass_corr'] = np.empty(shape=len(galaxy_list))
new_columns['sfr_flag'] = np.full_like(final_sample['nad_notch'], fill_value=True)
new_columns['sfr'] = np.empty(shape=len(final_sample))
new_columns['b/a'] = np.empty(shape=len(galaxy_list))
new_columns['distance'] = np.empty(shape=len(galaxy_list))
new_columns['kpc_scale'] = np.empty(shape=len(galaxy_list))
new_columns['ha_flux'] = np.empty(shape=len(galaxy_list))

for galaxy in galaxy_list:

    galaxy_index = np.argwhere(final_sample.index == galaxy)[0][0]

    galaxy_path = 'C:/Users/ariel/Workspace/GASP/High-z/SINOPSIS/' + galaxy + '/'

    sinopsis_cube = SinopsisCube(galaxy_path)   

    contours = fits.open(highz_dir + 'Data/Contours/' + galaxy + '_mask_ellipse.fits')[0].data
    g_band = fits.open(highz_dir + 'Data/G-Band/' + galaxy + '_DATACUBE_FINAL_v1_SDSS_g.fits')[1].data

    sinopsis_flag = sinopsis_cube.mask == 1
    contours_flag = contours > 3
    contours_and_sinopsis_flag = contours_flag & sinopsis_flag

    n_spaxels = sinopsis_flag.sum()
    n_disk = contours_flag.sum()
    n_spaxels_disk = contours_and_sinopsis_flag.sum()
    flux_spaxels = np.ma.masked_array(g_band, mask=~sinopsis_flag).sum()
    flux_disk = np.ma.masked_array(g_band, mask=~contours_flag).sum()
    flux_spaxels_disk = np.ma.masked_array(g_band, mask=~contours_and_sinopsis_flag).sum()
    
    new_columns['mass'][galaxy_index] = np.sum(sinopsis_cube.properties['TotMass2'])
    new_columns['mass_corr'][galaxy_index] = np.sum(sinopsis_cube.properties['TotMass2']) * (flux_disk/flux_spaxels_disk)

    ellipse_galaxy = ellipse_parameters[ellipse_parameters['name'] == galaxy]
    ellipse_ratio = ellipse_galaxy['B']/ellipse_galaxy['A'] 
    new_columns['b/a'][galaxy_index] = ellipse_ratio

    distance = redshift2lumdistance(final_sample['z'][galaxy_index], unit='cm')
    kpc_scale = 0.2 * arcsectokpc(final_sample['z'][galaxy_index]) 

    new_columns['distance'][galaxy_index] = distance
    new_columns['kpc_scale'][galaxy_index] = kpc_scale

    if sample['Ha_in'].to_list()[i]:

        try:
            bpt_map = fits.open(emission_dir + galaxy + '_bpt_classification_mario.fits')[0].data[0, :, :]
            bpt_map = np.ma.masked_array(bpt_map, np.isnan(bpt_map))

            emission_only = kv(highz_dir + 'Data/Emission/' + galaxy + '_DATACUBE_FINAL_v1_ec_eo_res_gau_spax_noflag.fits')
        except:
            new_columns['sfr_flag'][galaxy_index] = False
            continue

        ha_raw = emission_only.get_flux('Ha')
        sn_ha = emission_only.get_snr('Ha')
        sn_mask = (ha_raw <= 0) | (sn_ha < 2) | np.isnan(ha_raw) | np.isnan(sn_ha)| np.isinf(ha_raw) | np.isinf(sn_ha)
        
        mask = sn_mask | (contours < 5) | (bpt_map != 1)
        
        ha_flux_map = np.ma.masked_array(ha_raw, mask=mask) 
        ha_flux_total = np.sum(ha_flux_map[~ha_flux_map.mask])
        ha_luminosity_total = ha_flux_total * 1e-20 * 4 * np.pi * distance **2 / kpc_scale ** 2
        sfr = ha_luminosity_total / (10 ** 41.28)

        new_columns['sfr'][galaxy_index] = sfr

    else:
        continue

for new_column in new_columns.keys():
    final_sample[new_column] = new_columns[new_column]

final_sample.to_csv(data_dir + 'galaxy_sample.csv')

# flag = final_sample['sfr_flag'] & (final_sample['Category'] !='Post-Starburst')  & (final_sample['Category'] !='Quiescent') & (final_sample['Category'] != 'Truncated Disk') 

# sns.scatterplot(final_sample[flag], x='mass_corr', y='sfr', hue='Category')
# plt.xscale('log')
# plt.yscale('log')