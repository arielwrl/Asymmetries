import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data_dir = 'C:/Users/ariel/Workspace/GASP/High-z/PSB Asymetries/Data/'

sample = pd.read_table(data_dir + 'sample_v2.dat', delim_whitespace=True)

sample['RPS_class_code'] = sample['RPS_class'].copy()

sample['Dec'] = sample['DEC']
sample['Ha_in'] = sample['Ha_in'].astype(bool)

sample['RPS_class'][sample['RPS_class_code'] == 0] = 'Unaffected'
sample['RPS_class'][sample['RPS_class_code'] == 1] = 'Jellyfish'
sample['RPS_class'][sample['RPS_class_code'] == 2] = 'Post=Starburst'
sample['RPS_class'][sample['RPS_class_code'] == 3] = 'Truncated Disk'
sample['RPS_class'][sample['RPS_class_code'] < 0] = 'Unclassified'

sample['Category'] = np.full_like(sample['RPS_class'], fill_value='Pending')
sample['Category'][sample['RPS_class'] == 'Post=Starburst'] = 'Post=Starburst'
sample['Category'][sample['RPS_class'] == 'Truncated Disk'] = 'Truncated Disk'
sample['Category'][sample['RPS_class'] == 'Jellyfish'] = 'Jellyfish'
sample['Category'][(sample['RPS_class'] == 'Jellyfish') & sample['Memb'].astype(bool) & 
                sample['gas_disk'].astype(bool)] = 'Cluster Control'
sample['Category'][(sample['RPS_class'] == 'Jellyfish') & ~sample['Memb'].astype(bool) & 
                sample['gas_disk'].astype(bool)] = 'Field Control'

sample_flag = np.array([category in ['Post=Starburst', 'Truncated Disk', 'Jellyfish', 
                                      'Cluster Control', 'Field Control'] for 
                                      category in sample['Category']])

final_sample = sample[['ID', 'RA', 'Dec', 'z', 'Category', 'Ha_in']]
final_sample = final_sample[sample_flag]

final_sample.to_csv(data_dir + 'galaxy_sample.csv', index=False)