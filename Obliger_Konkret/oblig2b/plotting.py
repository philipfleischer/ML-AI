import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc


def plot_single_mnist_image(image, pred=None, label=None, show=True, title=None):
    """
    Display a single MNIST image.

    Arguments:
        image (np.array): [784,]-shaped array of mnist image.
        pred (int): Prediction corresponding to the image. If provided, will include the prediction in the title.
        label (int): label corresponding to the images. If given, will include label as the title to the image.
        show (bool): If True, will call plt.show().
        title (str): Title for the plot.
    """
    image = np.reshape(image, (28, 28))  # MNIST height and width
    plt.imshow(image, cmap="gray")
    plt.xticks(np.array([]))  # Remove ticks
    plt.yticks(np.array([]))
    if title is None:
        title = ""  # Construct title
    else:
        title += ". "
    if pred is not None:
        title += f"Prediction: {pred}"
    if pred is not None and label is not None:
        title += ", "
    if label is not None:
        title += f"True label: {label}"
    plt.title(title)
    if show:
        plt.show()


def plot_mnist_images(images, predictions=None, labels=None, show=True, n_images=10, n_cols=5, title=None):
    """
    Plot multiple MNIST images.

    Arguments:
        images (np.array): [n x 784]-shaped array of images.
        predictions (np.array): [n]-shaped array of predicted labels corresponding to the images.
            If provided, will include predictions as titles to the images.
        labels (np.array): [n]-shaped array of labels corresponding to the images.
            If provided, will include labels as titles to the images.
        show (bool): If True, will call plt.show().
        n_images (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns
        title (str): Title for the plot.
    """
    fig = plt.figure()
    n_rows = int(np.ceil(n_images / n_cols))
    for i in range(n_images):
        if predictions is not None and labels is not None:  # Make title blue in wrong predictions
            if predictions[i] != labels[i]:
                rc("text", color="blue")
            else:
                rc("text", color="black")

        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        image = images[i]
        image = np.reshape(image, (28, 28))  # Reshape to MNIST height and width
        ax.imshow(image, cmap="gray")  # Plot the image

        sub_title = ""  # Construct sub_title
        if predictions is not None:
            sub_title += f"P: {predictions[i]}"
        if predictions is not None and labels is not None:
            sub_title += ", "
        if labels is not None:
            sub_title += f"Y: {labels[i]}"

        ax.title.set_text(sub_title)

        plt.xticks(np.array([]))  # Remove ticks
        plt.yticks(np.array([]))

    rc("text", color="black")  # Set text back to black
    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    if show:
        plt.show()


def plot_random_mnist_images(
    images, predictions=None, labels=None, show=True, n_random=10, n_cols=5, title=None, seed=57
):
    """
    Plot random images from MNIST.

    Arguments:
        images (np.array): [n x 784]-shaped array of images.
        predictions (np.array): [n]-shaped array of predicted labels corresponding to the images.
            If provided, will include predictions as titles to the images.
        labels (np.array): [n]-shaped array of labels corresponding to the images.
            If provided, will include labels as titles to the images.
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns
        title (str): Title for the plot.
        seed (int): Seed for the random images to plot.
    """
    n = images.shape[0]
    np.random.seed(seed=seed)
    indices = np.random.choice(n, n_random, replace=False)  # Chose indices for random images to plot
    random_images = images[indices]
    if predictions is not None:
        predictions = predictions[indices]
    if labels is not None:
        labels = labels[indices]
    plot_mnist_images(
        images=random_images,
        predictions=predictions,
        labels=labels,
        show=show,
        n_images=n_random,
        n_cols=n_cols,
        title=title,
    )


def plot_mislabeled_mnist_images(images, predictions, labels, show=True, n_random=10, n_cols=5, title=None, seed=57):
    """
    Plot random mislabeled images from the data provided.

    Arguments:
        images (np.array): [n x 784]-shaped array of images.
        predictions (np.array): [n]-shaped array of predicted labels corresponding to the images.
        labels (np.array): [n]-shaped array of labels corresponding to the images.
        show (bool): If True, will call plt.show().
        n_random (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns.
        title (str): Title for the plot.
        seed (int): Seed for the random images to plot.
    """
    indices = predictions != labels
    mislabeled_images = images[indices]  # Index wrongly labeled images
    predictions = predictions[indices]
    labels = labels[indices]
    # Use random plotting function
    plot_random_mnist_images(
        images=mislabeled_images,
        predictions=predictions,
        labels=labels,
        show=show,
        n_random=n_random,
        n_cols=n_cols,
        title=title,
        seed=seed,
    )


def plot_worst_predicted_mnist_images(images, logits, labels, show=True, n_images=10, n_cols=5, title=None):
    """
    Plot (probably mislabeled) images that corresponds to the worst predictions. This means
    the value for the true class and the predicted logit value is as different as possible.

    Arguments:
        images (np.array): [n x 784]-shaped array of images.
        logits (np.array): [n]-shaped array of predicted logits values.
        labels (np.array): [n]-shaped array of labels corresponding to the images.
        show (bool): If True, will call plt.show().
        n_images (int): The amount of images that will be plotted.
        n_cols (int): The amount of images in each columns
        title (str): Title for the plot.
    """
    predicted_logits = logits[np.arange(len(labels)), labels]
    indices = predicted_logits.argsort()
    worst_images = images[indices[:n_images]]
    logits = logits[indices[:n_images]]
    labels = labels[indices[:n_images]]
    predictions = logits.argmax(axis=1)
    plot_mnist_images(
        images=worst_images,
        predictions=predictions,
        labels=labels,
        show=show,
        n_images=n_images,
        n_cols=n_cols,
        title=title,
    )
