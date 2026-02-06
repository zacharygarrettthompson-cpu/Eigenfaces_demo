from sklearn.datasets import fetch_lfw_people
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform

faces = fetch_lfw_people()

#####################################################################################################################################################
# alright now lets use eigenfaces to create a trippey ass photo of ms. nood
# we need to reshape my image into one of ms. nood
#A = imread("Cat.jpg")
#CAT = np.mean(A, axis=-1); # Convert RGB to grayscale
#CAT = CAT.reshape(62, 47)
#plt.axis('off')
#img = axi.imshow(CAT, cmap='gray')
CAT = io.imread("Zack.jpg")
#CAT = color.rgb2gray(CAT)
CAT = transform.resize(CAT, (62, 47), anti_aliasing=True)
#CAT = ax.flat
CAT = np.mean(CAT, axis=-1)
#axi.imshow(CAT, cmap='gray')
#CAT = ax.flat
#CAT.flatten()
n_eigenfaces = [20, 50, 100, 500]
fig, ax = plt.subplots(len(compressed_faces), 1 + len(n_eigenfaces), figsize=(15, 10))

for i in range(1):
    ax[i, 0].imshow(CAT, cmap='gray')
    ax[i, 0].set_title('original_image')

for j, n_eigenface in enumerate(n_eigenfaces):
    U_significant = U[:, :n_eigenface]

    compressed_faces = []
    #for i, face in enumerate(CAT.flat):
    compressed_faces.append(U_significant.T @ (CAT.flatten() - mean.squeeze()))
    reconstructed_faces = []
    for compressed_face in compressed_faces:
        reconstructed_faces.append((U_significant @ compressed_face + mean.squeeze()).reshape(62, 47))

    for i in range(1):
        ax[i, 1 + j].imshow(reconstructed_faces[i], cmap='gray')
        ax[i, 1 + j].set_title(f'decoded from r={n_eigenface}')
#plt.show()