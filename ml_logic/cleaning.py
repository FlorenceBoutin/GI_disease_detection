from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_images(folder):
    """
    Enter a folder directory with images to load.
    """
    datagen = ImageDataGenerator(rescale = 1. / 255)
    images = datagen.flow_from_directory(folder,
                                         target_size = (224, 224),
                                         color_mode = "rgb",
                                         batch_size = 32,
                                         class_mode = "categorical")
    return images

def train_val_test_generator(folder):
    """
    Generate the train, validation, and test batches.
    """
    #the directories need to be updated once we have the data in the cloud
    train_directory = "/home/emilyma/code/FlorenceBoutin/GI_disease_detection/raw_data/train"
    val_directory = "/home/emilyma/code/FlorenceBoutin/GI_disease_detection/raw_data/val"
    test_directory = "/home/emilyma/code/FlorenceBoutin/GI_disease_detection/raw_data/test"

    train_dataset = load_images(train_directory)
    val_dataset = load_images(val_directory)
    test_dataset = load_images(test_directory)

    return train_dataset, val_dataset, test_dataset
