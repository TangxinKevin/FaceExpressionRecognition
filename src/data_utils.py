from sklearn.model_selection import train_test_split

def split_dataset(images, labels, validation_split):
    """
    Split dataset into train set and validation set.

    # param:
        images: numpy array of image data or list 
        labels: numpy arrary of one-hot vector labels
    """
    train_images, test_images, train_labels, test_labels = train_test_split(images,
        labels, test_size=validation_split, random_state=42, stratify=labels)
    return (train_images, train_labels), (test_images, test_labels)
    