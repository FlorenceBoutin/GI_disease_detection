import os
import cv2

def load_images(folder):
    """
    Enter a folder directory with images to load.
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img_rgb is not None:
            images.append(img_rgb)
    return images

def resize_images(images):
    """
    Resize the images to size (224, 224, 3) and normalize by dividing by 255.
    """
    final_images = []
    for image in images:
        resized_image = cv2.resize(image, (224, 224))
        normalized_image = resized_image / 255
        final_images.append(normalized_image)
    return final_images

def create_X_y(images):
    """
    From the original dataset, define the X and y.
    X: cleaned images for normal, ulcerative colitis, and polyps
    y: classification for normal (0), ulcerative colitis (1), and polyps (2)
    """
    pass

def clean_images(folder):
    images = load_images(folder)
    resized_images = resize_images(images)
    X, y = create_X_y(resized_images)

    return X, y
