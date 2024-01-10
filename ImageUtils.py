import numpy as np


def parse_record(record: np.ndarray, training: bool) -> np.ndarray:
    """
    Parse and preprocess a record for image data.
    Args:
        record (numpy.ndarray): The image record to be parsed.
        training (bool): Flag indicating whether the record is for training.
    Returns:
        numpy.ndarray: Preprocessed image.
    """
    if record.shape != (32, 32, 3):
        depth_major = record.reshape((3, 32, 32))
        record = np.transpose(depth_major, [1, 2, 0])
    image = preprocess_image(record, training)

    return image


def preprocess_image(image: np.ndarray, training: bool) -> np.ndarray:
    """
    Preprocess the image for training or testing.
    Args:
        image (np.ndarray): The image to preprocess.
        training (bool): Flag indicating whether preprocessing for training.
    Returns:
        np.ndarray: Preprocessed image.
    """
    if training:
        # image = tf.image.resize_image_with_crop_or_pad(image, 32 + 8, 32 + 8)
        padded_image = np.zeros(
            (image.shape[0] + 8, image.shape[1] + 8, image.shape[2])
        )
        padded_image[4:4 + image.shape[0], 4:4 + image.shape[1], :] = image
        image = padded_image

        # Randomly crop a [32, 32] section of the image.
        # image = tf.random_crop(image, [32, 32, 3])
        uppre_lefts = np.random.randint(low=1, high=8, size=2)
        image = image[
            uppre_lefts[0]: uppre_lefts[0] + 32,
            uppre_lefts[1]: uppre_lefts[1] + 32,
            :,
        ]

        # image = tf.image.random_flip_left_right(image)
        if np.random.random() > 0.5:
            image = np.flip(image, 1)

    # image = tf.image.per_image_standardization(image)
    image = (image - np.mean(image)) / (np.std(image) + np.random.random())

    return image
