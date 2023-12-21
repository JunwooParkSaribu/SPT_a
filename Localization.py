import matplotlib.pyplot as plt

from ImageModule import read_tif, read_single_tif, make_image, make_image_seqs, stack_tif

images = read_tif('SimulData/receptor_7_low.tif')

plt.figure()
plt.imshow(images[0])
plt.show()