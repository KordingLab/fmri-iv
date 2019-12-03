from os import path
import requests

# Mask labels
mask_filename = 'aal_space-MNI152NLin6_res-2x2x2.json'
mask_path = '../data_bnu2/' + mask_filename

if not path.isfile(mask_path):
    mask_url = 'https://github.com/neurodata/neuroparc/blob/master/atlases/label/Human/' + mask_filename + '?raw=true'
    mask_labels = requests.get(mask_url)
    with open(mask_path, 'wb') as mask_file:
        mask_file.write(mask_labels.content)

# ROI timeseries
