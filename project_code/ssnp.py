"""
Class for decision boundary visualization using the "Supervised Decision Boundary Maps" (SDBM) method.
Parts of the code is adapted from the following studies:

SSNP paper: "Self-supervised Dimensionality Reduction with Neural Networks and Pseudo-labeling", 
    by M. Espadoto, N. S. T. Hirata and A. C. Telea, presented at IVAPP 2021.
    URL: https://github.com/mespadoto/ssnp

SDBM paper: "SDBM: supervised decision boundary maps for machine learning classifiers",
    by A. A. A. M. Oliveira, M. Espadoto, R. Hirata Jr. and A. C. Telea, presented at VISIGRAPP 2022.
    URL: https://github.com/mespadoto/sdbm
"""

# Importing necessary packages, and classes adapted from decision boundary visualization studies

import matplotlib.cm as cm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from skimage.color import rgb2hsv, hsv2rgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import cartesian
import tensorflow as tf
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import torch.nn as nn
import Models

import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

# function to save visualizations as png files
# adapted from https://github.com/mespadoto/sdbm
def results_to_png( 
        np_matrix, 
        prob_matrix, 
        grid_size, 
        n_classes,
        dataset_name, 
        classifier_name, 
        output_dir, 
        real_points=None,
        adversarial_points=None,
        max_value_hsv=None,
        suffix=None
        ):

    if suffix is not None:
        suffix = f"_{suffix}"
    else:
        suffix = ""
    data = cm.tab20(np_matrix/n_classes)
    data_vanilla = data[:,:,:3].copy()

    if max_value_hsv is not None:
        data_vanilla = rgb2hsv(data_vanilla)
        data_vanilla[:, :, 2] = max_value_hsv
        data_vanilla = hsv2rgb(data_vanilla)
    
    if real_points is not None:
        data_vanilla = rgb2hsv(data_vanilla)
        print(len(data_vanilla[real_points[:, 0], real_points[:, 1], 2]))
        data_vanilla[real_points[:, 0], real_points[:, 1], 2] = 1
        data_vanilla = hsv2rgb(data_vanilla)

    if adversarial_points is not None:
    #if False:
        data_vanilla = rgb2hsv(data_vanilla)
        print(len(data_vanilla[adversarial_points[:, 0], adversarial_points[:, 1], 2]))
        data_vanilla[adversarial_points[:, 0], adversarial_points[:, 1], 2] = 0
        #data_vanilla[adversarial_points[:, 0], adversarial_points[:, 1], 0] = 0.0
        data_vanilla = hsv2rgb(data_vanilla)
    
    data_alpha = data.copy()

    data_hsv = data[:,:,:3].copy()
    data_alpha[:,:,3] = prob_matrix

    data_hsv = rgb2hsv(data_hsv)
    data_hsv[:,:,2] = prob_matrix
    data_hsv = hsv2rgb(data_hsv)

    rescaled_vanilla = (data_vanilla*255.0).astype(np.uint8)
    im_vanilla = Image.fromarray(rescaled_vanilla)
    print(f"Saving vanilla. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    im_vanilla.save(os.path.join(output_dir,f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_vanilla{suffix}.png"))

    rescaled_alpha = (255.0*data_alpha).astype(np.uint8)
    im_alpha = Image.fromarray(rescaled_alpha)
    print(f"Saving alpha. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    im_alpha.save(os.path.join(output_dir,f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_alpha{suffix}.png"))

    rescaled_hsv = (255.0*data_hsv).astype(np.uint8)
    im_hsv = Image.fromarray(rescaled_hsv)
    print(f"Saving hsv. {grid_size}x{grid_size} - {dataset_name} - {classifier_name}")
    im_hsv.save(os.path.join(output_dir,f"{classifier_name}_{grid_size}x{grid_size}_{dataset_name}_hsv{suffix}.png"))

    # Add a legend for the colors and labels
    legend_elements = [
        Patch(facecolor=cm.tab20(i / n_classes)[:3], label=f"{i}")
        for i in range(n_classes)
    ]

    # Create a figure for the legend
    plt.figure(figsize=(6, 1))
    plt.legend(handles=legend_elements, loc='center', ncol=n_classes, frameon=False)
    plt.axis('off')
    plt.title("Legend")
    plt.tight_layout()
    plt.show()
    
    return im_vanilla, im_alpha, im_hsv

# SSNP model class
# Code adapted from: https://github.com/mespadoto/ssnp/blob/main/code/ssnp.py
class SSNP():
    
    def __init__(self, init_labels='precomputed', epochs=100,
            input_l1=0.0, input_l2=0.0, bottleneck_l1=0.0,
            bottleneck_l2=0.5, verbose=1, opt='adam',
            bottleneck_activation='tanh', act='relu',
            init='glorot_uniform', bias=0.0001, patience=3,
            min_delta=0.01):
        self.init_labels = init_labels
        self.epochs = epochs
        self.verbose = verbose
        self.opt = opt
        self.act = act
        self.init = init
        self.bias = bias
        self.input_l1 = input_l1
        self.input_l2 = input_l2
        self.bottleneck_l1 = bottleneck_l1
        self.bottleneck_l2 = bottleneck_l2
        self.bottleneck_activation = bottleneck_activation
        self.patience = patience
        self.min_delta = min_delta

        self.label_bin = LabelBinarizer()

        self.model = None
        self.fwd = None
        self.inv = None

        tf.random.set_seed(42)

        self.is_fitted = False
        K.clear_session()

    def save_model(self, saved_model_folder):
        tf.keras.models.save_model(self.model, saved_model_folder)

    def load_model(self, saved_model_folder):
        self.model = tf.keras.models.load_model(saved_model_folder)


        self.model.compile(optimizer=self.opt,
                    loss={'main_output': 'categorical_crossentropy', 'decoder_output': 'binary_crossentropy'},
                    metrics=['accuracy', 'mse'])
        
        model = self.model

        main_input = model.inputs
        main_output = model.get_layer('main_output')
        encoded = model.get_layer('encoded')


        encoded_input = Input(shape=(2,))
        l = model.get_layer('enc1')(encoded_input)
        l = model.get_layer('enc2')(l)
        l = model.get_layer('enc3')(l)
        decoder_layer = model.get_layer('decoder_output')(l)

        self.inv = Model(encoded_input, decoder_layer)

        self.fwd = Model(inputs=main_input, outputs=encoded.output)
        self.clustering = Model(inputs=main_input, outputs=main_output.output)

        self.is_fitted = True


    def fit(self, X, y=None):
        if y is None and self.init_labels == 'precomputed':
            raise Exception('Must provide labels when using init_labels = precomputed')
        
        if y is None:
            y = self.init_labels.fit_predict(X)

        self.label_bin.fit(y)

        main_input = Input(shape=(X.shape[1],), name='main_input')
        x = Dense(512,  activation=self.act,
                        kernel_initializer=self.init,
                        bias_initializer=Constant(self.bias))(main_input)
        x = Dense(128,  activation=self.act,
                        kernel_initializer=self.init,
                        bias_initializer=Constant(self.bias))(x)
        x = Dense(32, activation=self.act,
                        activity_regularizer=regularizers.l1_l2(l1=self.input_l1, l2=self.input_l2),
                        kernel_initializer=self.init,
                        bias_initializer=Constant(self.bias))(x)
        encoded = Dense(2,
                        activation=self.bottleneck_activation,
                        kernel_regularizer=regularizers.l1_l2(l1=self.bottleneck_l1, l2=self.bottleneck_l2),
                        kernel_initializer=self.init,
                        name='encoded',
                        bias_initializer=Constant(self.bias))(x)

        x = Dense(32, activation=self.act, kernel_initializer=self.init, name='enc1', bias_initializer=Constant(self.bias))(encoded)
        x = Dense(128, activation=self.act, kernel_initializer=self.init, name='enc2', bias_initializer=Constant(self.bias))(x)
        x = Dense(512, activation=self.act, kernel_initializer=self.init, name='enc3', bias_initializer=Constant(self.bias))(x)

        n_classes = len(np.unique(y))
        
        if n_classes == 2:
            n_units = 1
            main_output_activation = 'sigmoid'
            main_loss = 'binary_crossentropy'
        else:
            n_units = n_classes
            main_output_activation = 'softmax'
            main_loss = 'categorical_crossentropy'

        main_output = Dense(n_units,
                            activation=main_output_activation,
                            name='main_output',
                            kernel_initializer=self.init,
                            bias_initializer=Constant(self.bias))(x)

        decoder_output = Dense( X.shape[1],
                                activation='sigmoid',
                                name='decoder_output',
                                kernel_initializer=self.init,
                                bias_initializer=Constant(self.bias))(x)

        model = Model(inputs=main_input, outputs=[main_output, decoder_output])
        self.model = model 

        model.compile(optimizer=self.opt,
                    loss={'main_output': main_loss, 'decoder_output': 'binary_crossentropy'},
                    metrics=['accuracy', 'mse'])

        if self.patience > 0:
            callbacks = [EarlyStopping(monitor='val_loss', mode='min', min_delta=self.min_delta, patience=self.patience, restore_best_weights=True, verbose=self.verbose)]
        else:
            callbacks = []

        

        hist = model.fit(X,
                    [self.label_bin.transform(y), X],
                    batch_size=32,
                    epochs=self.epochs,
                    shuffle=True,
                    verbose=self.verbose,
                    validation_split=0.05,
                    callbacks=callbacks)

        encoded_input = Input(shape=(2,))
        l = model.get_layer('enc1')(encoded_input)
        l = model.get_layer('enc2')(l)
        l = model.get_layer('enc3')(l)
        decoder_layer = model.get_layer('decoder_output')(l)

        self.inv = Model(encoded_input, decoder_layer)

        self.fwd = Model(inputs=main_input, outputs=encoded)
        self.clustering = Model(inputs=main_input, outputs=main_output)
        self.is_fitted = True

        return hist

    def transform(self, X):
        if self._is_fit():
            return self.fwd.predict(X)
           
    def inverse_transform(self, X_2d):
        if self._is_fit():
            return self.inv.predict(X_2d)

    def predict(self, X):
        if self._is_fit():
            y_pred = self.clustering.predict(X)
            return self.label_bin.inverse_transform(y_pred)

    def _is_fit(self):
        if self.is_fitted:
            return True
        else:
            raise Exception('Model not trained. Call fit() before calling transform()')

def visualize_decision_boundaries(
        original_dataset: torch.utils.data.Dataset,
        dataset_name: str,
        classifier_model: nn.Module,
        classifier_model_name: str,
        ssnp_path_and_name: str,        
        image_output_path: str = "../output/images",
        image_grid_size: int = 300,
        batch_size: int = 100000,
        adversarial_images: torch.Tensor = None,
        ssnp_training_epochs: int = 50,
        ssnp_training_patience: int = 10,
        device: str = "cpu",
        verbose: bool = False,
)-> tuple[SSNP, np.ndarray, np.ndarray] :
    """
    Produces three image types (Vanilla, Alpha, and HSV) for the decision boundaries of a classifier.
    Saves these images in the specified file path as .png, and prints them to the terminal.

    Args:
        original_dataset (torch.utils.data.Dataset): The original data set from which adversarial
            examples are produced. Used for the training of the SSNP dimensionality reduction model.
        dataset_name (str): A name given to the original dataset.
        classifier_model (nn.Module): The classification model for which to produce decision boundaries.
        classifier_model_name (str): A name given to the classification model.
        ssnp_path_and_name (str): The path (relative to the cwd) and name (without the file extension) of the SSNP.
            dimensionality reduction model. If the path and name already exist, the SSNP is imported from
            this path, otherwise it is trained from scratch and saved at the path.
        image_output_path (str): Location to which the images produced by the visualization should be saved, relative
            to the cwd. Defaults to "../output/images".
        image_grid_size (int): The width and height of the decision boundary visualization square, in pixels.
            Image created is of size image_grid_size x image_grid_size. Defaults to 300.
        batch_size (int): The maximum number of images to project from the 2D grid space to the image space at a time.
            Specify if memory constraints are a concern. Defaults to 100,000 images.
        adversarial_images (torch.Tensor): Torch tensor of size [B, C, W, H] which represent adversarial images to
            explicitly display on the visualization grid, in addition to the original data points. Defaults to None.
        ssnp_training_epochs (int): The maximum number of epochs to train the SSNP model. Defaults to 50.
        ssnp_training_patience (int): The number of epochs which the SSNP training will wait without performance
            improvements before stopping prematurely. None defaults to 10.
        device (str): The device on which to perform all the tensor operations. Accepts ["gpu", "mps", "cpu"]. 
            Defaults to "cpu".
        verbose (bool): Whether to print status updates and the final images to the terminal. Defaults to False.

    Returns
    -------
        ssnp : SSNP
            The SSNP dimensionality reduction model trained or imported by this function and used for the visualization. Outputted for convenience so that further tests and analysis can be performed using the forward (image space [C, W, H] -> grid space [x, y]) or inverse (grid space -> image space) projection on your choice of data.
        im_grid : ndarray
            A numpy array containing the predicted labels for each pixel in the grid.
        prob_grid : ndarray
            A numpy array containing the prediction confidence for the predicted label, for each pixel in the grid.
    """

    ### ORIGINAL TRAINING DATASET ###

    # splitting data into X and Y:
    x = original_dataset.data.numpy()
    y = original_dataset.targets.numpy()

    # flatten the image data and get the number of channels
    x_flattened = x.reshape(x.shape[0], -1)
    image_shape = x.shape[1:]
    image_channels = 1 if len(image_shape) == 2 else image_shape[0]

    ### ADVERSARIAL DATA ###

    # if adversarial images are provided, convert them to numpy, flatten them and normalize
    if adversarial_images is not None:
        adversarial_x = adversarial_images.to(device).numpy()
        adversarial_x_flattened = np.reshape(adversarial_x, (adversarial_x.shape[0], -1))

        # if there are adversarial images, append them to the original dataset so that the normalization accounts for all data        
        all_x_flattened = np.concatenate((x_flattened, adversarial_x_flattened), axis=0)

    else:
        all_x_flattened = x_flattened

    # initialize the scaler and fit it to the combined data
    scaler = MinMaxScaler()
    scaler.fit(all_x_flattened)

    # normalize the original data
    normalized_x = scaler.transform(x_flattened)
    normalized_x = normalized_x.astype('float32')

    # normalize the adversarial data if provided
    if adversarial_images is not None:
        normalized_adversarial_x = scaler.transform(adversarial_x_flattened)
        normalized_adversarial_x = normalized_adversarial_x.astype('float32')

    ### CLASSIFIER MODEL ###    

    # set the classifier to evaluation mode after checking the device
    model_device = next(classifier_model.parameters()).device
    assert str(model_device) == device, f"Model device {model_device} does not match specified device {device}."
    classifier_model.eval()

    ### SSNP MODEL ###

    # select ssnp model device
    if device == "gpu": device_tf = '/GPU:0'
    elif device == "mps": device_tf = '/MPS:0'
    elif device == "cpu": device_tf = '/CPU:0'
    else: raise ValueError("Invalid device specified. Choose from ['gpu', 'mps', 'cpu'].")

    # check if the SSNP model already exists at given path
    ssnp_model_dir = ssnp_path_and_name + ".keras"
    if os.path.exists(ssnp_model_dir): # load the model from preexisting path
        ssnp = SSNP(
                epochs=ssnp_training_epochs, 
                verbose=verbose, 
                patience=ssnp_training_patience, 
                opt='adam', 
                bottleneck_activation='linear'
                )
        ssnp.load_model(ssnp_model_dir)
    else: # train the model and save it to the given path
        with tf.device(device_tf):
            ssnp = SSNP(
                    epochs=ssnp_training_epochs, 
                    verbose=verbose, 
                    patience=ssnp_training_patience, 
                    opt='adam', 
                    bottleneck_activation='linear'
                    )
            ssnp.fit(normalized_x, y)
            ssnp.save_model(ssnp_model_dir)

    ### IMAGE DATA PROJECTION ###

    # using the newly trained SSNP model, project the images to the 2D space
    two_dim_projected_original_x = ssnp.transform(normalized_x)
    if adversarial_images is not None: two_dim_projected_adversarial_x = ssnp.transform(normalized_adversarial_x)

    # combine the original and adversarial data projections
    if adversarial_images is not None:
        two_dim_projected_x = np.concatenate((two_dim_projected_original_x, two_dim_projected_adversarial_x), axis=0)
    else:
        two_dim_projected_x = two_dim_projected_original_x

    ### 2D GRID SPACE ###

    # get min and max coordinates of the projected data
    scaler = MinMaxScaler()
    scaler.fit(two_dim_projected_x)
    xmin = np.min(two_dim_projected_x[:, 0])
    xmax = np.max(two_dim_projected_x[:, 0])
    ymin = np.min(two_dim_projected_x[:, 1])
    ymax = np.max(two_dim_projected_x[:, 1])

    # initialize 2D arrays for the class and probability predictions
    img_grid = np.zeros((image_grid_size,image_grid_size))
    prob_grid = np.zeros((image_grid_size,image_grid_size))

    # create a grid of points in the 2D space
    x_intrvls = np.linspace(xmin, xmax, num=image_grid_size)
    y_intrvls = np.linspace(ymin, ymax, num=image_grid_size)

    x_grid = np.linspace(0, image_grid_size-1, num=image_grid_size)
    y_grid = np.linspace(0, image_grid_size-1, num=image_grid_size)

    pts = cartesian((x_intrvls, y_intrvls))
    pts_grid = cartesian((x_grid, y_grid))
    pts_grid = pts_grid.astype(int)

    # normalize the projected images to fit the grid
    scaler.fit(two_dim_projected_x)
    normalized_two_dim_original_x = scaler.transform(two_dim_projected_original_x)
    normalized_two_dim_original_x = normalized_two_dim_original_x.astype('float32')
    normalized_two_dim_original_x *= (image_grid_size-1)
    normalized_two_dim_original_x = normalized_two_dim_original_x.astype(int)

    if adversarial_images is not None:
        normalized_two_dim_adversarial_x = scaler.transform(two_dim_projected_adversarial_x)
        normalized_two_dim_adversarial_x = normalized_two_dim_adversarial_x.astype('float32')
        normalized_two_dim_adversarial_x *= (image_grid_size-1)
        normalized_two_dim_adversarial_x = normalized_two_dim_adversarial_x.astype(int)
    else: normalized_two_dim_adversarial_x = None
    
    ### CLASSIFICATION PREDICTIONS ###

    # process the grid points in batches due to memory constraints
    position = 0
    while position < len(pts):
        # extract batch of points from the grid
        pts_batch = pts[position : position + batch_size]

        # transform the points from 2D to original image space using inverse SSNP
        with tf.device(device_tf):
            image_batch = ssnp.inverse_transform(pts_batch)

        # predict the labels for synthetic points using the classifier
        with torch.no_grad():
            # unflatten the image to original shape
            image_batch_tensor = torch.tensor(image_batch, dtype=torch.float32)
            image_batch_tensor = image_batch_tensor.to(device)
            image_batch_tensor = image_batch_tensor.view(image_batch_tensor.shape[0], image_channels, image_shape[0], image_shape[1])
            image_batch_tensor = image_batch_tensor.to(device)
            
            # make predictions
            logits = classifier_model(image_batch_tensor)
            probs = F.softmax(logits, dim=1)
            labels = torch.argmax(probs, dim=1).cpu().numpy()
            alpha = torch.amax(probs, dim=1).cpu().numpy()

        # get new batch of points from the grid
        pts_grid_batch = pts_grid[position:position+batch_size]

        # advance position
        position += batch_size

        # assign the predicted labels and probabilities to the grid
        img_grid[
            pts_grid_batch[:, 0],
            pts_grid_batch[:, 1]
            ] = labels

        prob_grid[
            pts_grid_batch[:, 0],
            pts_grid_batch[:, 1]
            ] = alpha
        
    ### IMAGE GENERATION ###

    # get the number of classes in the dataset
    n_classes = len(np.unique(y))

    # Generate image from the predictions
    im_vanilla, im_alpha, im_hsv = results_to_png(
        np_matrix=img_grid,
        prob_matrix=prob_grid,
        grid_size=image_grid_size,
        dataset_name=dataset_name,
        classifier_name=classifier_model_name,
        output_dir=image_output_path,
        real_points=normalized_two_dim_original_x,
        adversarial_points=normalized_two_dim_adversarial_x,
        n_classes=n_classes)
    
    # print the images to the terminal
    if verbose:
        # Display the images in a row with labels
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create 1 row and 3 columns of subplots

        # Titles for the images
        titles = ["Vanilla", "Alpha", "HSV"]

        # Images to display
        images = [im_vanilla, im_alpha, im_hsv]

        # Loop through the images and axes
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)  # Display the image
            ax.set_title(title)  # Set the title
            ax.axis('off')  # Turn off the axis

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.show()

    return ssnp, img_grid, prob_grid