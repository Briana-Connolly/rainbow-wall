import cv2
import numpy as np
import colorsys

def loadPokemonImages(a,b):
    images = []
    for x in range(a,b):
        path = "images/" + str(x) + ".png"
        img = cv2.imread(path)
        scale_percent = 25 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        images.append(resized)
    return images

def calculateDominantColor(image):
    pixels = np.float32(image.reshape(-1, 3))
    background_color = [0,0,0]
    mask = np.all(pixels != background_color, axis = -1)
    pixels = pixels[mask]

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]
    dominant = colorsys.rgb_to_hsv(dominant[2], dominant[1], dominant[0])
    return dominant

def concatenateImages(images, rows=2, cols=2):
    rowImages = []
    for x in range(rows):
        rowImage = images[0 + x * cols]
        for y in range(1, cols):
            rowImage = np.concatenate((rowImage, images[y + x * cols]),axis=1)
        rowImages.append(rowImage)
    dispImage = rowImages[0]
    for x in range(1, len(rowImages)):
        dispImage = np.concatenate((dispImage, rowImages[x]),axis=0)
    return dispImage

images = loadPokemonImages(1,151)
sortedImages = sorted(images, key=lambda x: calculateDominantColor(x))
output = concatenateImages(sortedImages, 12, 12)

cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
