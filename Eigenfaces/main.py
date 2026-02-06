from sklearn.datasets import fetch_lfw_people
from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform

faces = fetch_lfw_people()

#plt.imshow(faces.images[0], cmap='gray')
#plt.show()
unique_face_images = []
#zack = imread('img_2.png')
#cat = imread('img_1.png')
held_out_faces = []
#held_out_faces = [zack, cat]
#held_out_faces = np.mean(held_out_faces, axis=-1)
#held_out_faces[0] = transform.resize(held_out_faces[0], (62, 47), anti_aliasing=True)
#held_out_faces[1] = transform.resize(held_out_faces[1], (62, 47), anti_aliasing=True)
seen_names = []
# importing photos for data set and verifying
i = 0
while len(unique_face_images) < 1000:
    if not faces.target_names[faces.target[i]] in seen_names:
        unique_face_images.append(faces.images[i])
        seen_names.append(faces.target_names[faces.target[i]])

    i += 1
while len(held_out_faces) < 2:
    if not faces.target_names[faces.target[i]] in seen_names:
        held_out_faces.append(faces.images[i])
        seen_names.append(faces.target_names[faces.target[i]])
        print(faces.target_names[faces.target[i]])
        #print(seen_names[i])
    i += 1
#print(seen_names[i-2])
#print(seen_names[i-1])
fig, ax = plt.subplots(5, 8, figsize=(12, 10))
for i, axi in enumerate(ax.flat):
    axi.imshow(unique_face_images[i], cmap='gray')
plt.show()
fig, ax = plt.subplots(1, 2, figsize=(12, 10))
for i, axi in enumerate(ax.flat):
    axi.imshow(held_out_faces[i], cmap='gray')
plt.show()
# constructing egienfaces
X = np.stack([face.flatten() for face in unique_face_images], axis=1)

print(X.shape)
# matrix mean operation
mean = np.mean(X, axis=1, keepdims=True)
U, Sigma, Vt = np.linalg.svd(X - mean)

# output check
#print('U shape: ', U.shape)
#print('Sigma shape: ', Sigma.shape)
#print('Vt shape: ', Vt.shape)
# rectangular output = bad we want square because we're squares
# using square matrixes instead.
U, Sigma, Vt = np.linalg.svd(X - mean, full_matrices=False)

print('U shape: ', U.shape)
print('Sigma shape: ', Sigma.shape)
print('Vt shape: ', Vt.shape)
# verify that matrix decomp represents orig data set
print(np.max(X - mean - U@np.diag(Sigma)@Vt))
# look at log plot to understand things
plt.plot(Sigma)
plt.title('Photo number vs. singular value size, log scale\n this shows that the vast majority of the data in the set is the same')
plt.yscale('log')
plt.xlabel("Photo number")
plt.ylabel("Singular values data size")
#plt.show()
# this plot shows that the majority of the data is the same
# Time to actually make egienfaces now
n_components = 40
U_significant = U[:,:n_components]

eigenfaces = [U_significant[:,i].reshape((62,47)) for i in range(n_components)]

fig, ax = plt.subplots(5, 8, figsize=(12, 10))
for i, axi in enumerate(ax.flat):
    axi.imshow(eigenfaces[i], cmap='gray')
#plt.title('Egienfaces\ndata significance decreases as we move further to the right and down\n As we move further down the chart more distinct features start to appear like eyes, eyebrows and crease lines')
#plt.show()
# significance means the most common data points
#Egienfaces\n data significance decreases as we move further to the right and down\n As we move further down the chart more distinct features start to appear like eyes, eyebrows and crease lines'



# Reconstructing Faces with the help of egienfaces 40x
compressed_faces = []
for i, face in enumerate(held_out_faces):
    compressed_faces.append(U_significant.T@(face.flatten() - mean.squeeze()))
    print(f'held out face {i}: ', compressed_faces[-1])
print(mean.squeeze())
reconstructed_faces = []
for compressed_face in compressed_faces:
    reconstructed_faces.append((U_significant @ compressed_face + mean.squeeze()).reshape(62, 47))

fig, ax = plt.subplots(len(compressed_faces), 2, figsize=(12, 10))

for i, (reconstructed_face, true_face) in enumerate(zip(reconstructed_faces, held_out_faces)):
    ax[i, 0].imshow(true_face, cmap='gray')
    ax[i, 1].imshow(reconstructed_face, cmap='gray')
plt.show()


# Eigenfaces Demo, basically shows how increasing the size of the dataset for the compression algorithm
# allows it to more or less increase the resolution of the regenerated image.
# This is because each face provides information on less significant details
# this allows us to have a similar amount of resolution with less then half the amount of data
# because it is using the data set to determine what is significant about a human face
# basically whatever features are the most common in a face tend to be the most important
# i wonder how this compares to typical
n_eigenfaces = [40, 100, 200, 500, 1000, 1000]
fig, ax = plt.subplots(len(compressed_faces), 1 + len(n_eigenfaces), figsize=(15, 10))

for i in range(2):
    ax[i, 0].imshow(held_out_faces[i], cmap='gray')
    ax[i, 0].set_title('original_image')

for j, n_eigenface in enumerate(n_eigenfaces):
    U_significant = U[:, :n_eigenface]

    compressed_faces = []
    for i, face in enumerate(held_out_faces):
        compressed_faces.append(U_significant.T @ (face.flatten() - mean.squeeze()))

    reconstructed_faces = []
    for compressed_face in compressed_faces:
        reconstructed_faces.append((U_significant @ compressed_face + mean.squeeze()).reshape(62, 47))

    for i in range(2):
        ax[i, 1 + j].imshow(reconstructed_faces[i], cmap='gray')
        ax[i, 1 + j].set_title(f'decoded from r={n_eigenface}')
plt.show()


########################################################################################################################################################
################ Change these File calls to get different images on the final plot.
zack = imread('Homework.jpg')
cat = imread('Cloud.jpg')
#############################################################################################################################################################
ZackECat_Raw = [zack, cat]
ZackECat_Raw = np.mean(ZackECat_Raw, axis=-1)
#ZackECat_Raw[0] = transform.resize(ZackECat_Raw[0], (62, 47), anti_aliasing=True)
#ZackECat_Raw[1] = transform.resize(ZackECat_Raw[1], (62, 47), anti_aliasing=True)

#fig, ax = plt.subplots(len(ZackECat_Raw), 2, figsize=(12, 10))
ZackECat_compressed = []
#ZackECat_Raw = []
#n_eigenfaces = [40, 100, 200, 500, 1000,]
#fig, ax = plt.subplots(len(ZackECat_Raw), 1 + len(n_eigenfaces), figsize=(15, 10))
for i, face in enumerate(ZackECat_Raw):
    ZackECat_compressed.append(U_significant.T@(face.flatten() - mean.squeeze()))
    print(f'held out face {i}: ', ZackECat_compressed[-1])
print(mean.squeeze())
#reconstructed_faces = []
ZackECat_Recon = []
for turd in ZackECat_compressed:
    ZackECat_Recon.append((U_significant @ turd + mean.squeeze()).reshape(62, 47))

#for i, (reconstructed_face, true_face) in enumerate(zip(ZackECat_Recon, ZackECat_Raw)):
#    ax[i, 0].imshow(true_face, cmap='gray')
#    ax[i, 1].imshow(reconstructed_face, cmap='gray')
#plt.show()

##################################################################################################################################################
n_eigenfaces = [40, 100, 150, 250, 400, 1000]
fig, ax = plt.subplots(len(ZackECat_compressed), 1 + len(n_eigenfaces), figsize=(15, 10))
for i in range(2):
    ax[i, 0].imshow(ZackECat_Raw[i], cmap='gray')
    ax[i, 0].set_title('original_image')

for j, n_eigenface in enumerate(n_eigenfaces):
    U_significant = U[:, :n_eigenface]

    ZackECat_compressed = []
    for i, face in enumerate(ZackECat_Raw):
        ZackECat_compressed.append(U_significant.T @ (face.flatten() - mean.squeeze()))

    ZackECat_Recon = []
    for compressed_face in ZackECat_compressed:
        ZackECat_Recon.append((U_significant @ compressed_face + mean.squeeze()).reshape(62, 47))

    for i in range(2):
        ax[i, 1 + j].imshow(ZackECat_Recon[i], cmap='gray')
        ax[i, 1 + j].set_title(f'decoded from r={n_eigenface}')
plt.show()
