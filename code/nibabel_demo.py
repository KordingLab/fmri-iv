import nibabel as nib
import matplotlib.pyplot as plt

epi_img = nib.load('../data/someones_epi.nii.gz')
epi_img_data = epi_img.get_fdata()
print('Shape of my data:', epi_img_data.shape)


def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, s in enumerate(slices):
        axes[i].imshow(s.T, cmap="gray", origin="lower")


slice_0 = epi_img_data[26, :, :]
slice_1 = epi_img_data[:, 30, :]
slice_2 = epi_img_data[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")
plt.show()
