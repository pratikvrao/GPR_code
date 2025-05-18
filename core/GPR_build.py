import numpy as np

# Standard GPR packages
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Chained GPR packages
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
from sklearn.cluster import KMeans


def GPR_fit(X_train, y_train, X_test):
    
    """
    Function to fit the standard GPR model to the training data;
    and predict the output for the test data.
    
    Args:
        (Ideally pandas dataframe)
        X_train: Input from the training dataset (feature values) 
        y_train: Ouput from the training dataset (target values)
        X_test:  Input from the test dataset (feature values for test set)
    
    Return:
        (np array)
        y_pred_test : Predicted output Y for the test dataset
    """
#     n_length_scales = len(X_train.columns)
    n_length_scales = X_train.shape[1]
    
    # Modify the kernel length scale, length scale bounds, noise level and noise level bounds based on your data.
    
    kernel = 1*RBF(length_scale = n_length_scales*[0.5]) \
             + WhiteKernel(noise_level = 0.5)
    
    model = gp.GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 9)
    
    
    # Fit
    model.fit(X_train, y_train)
    

    # Predict
    y_pred_test, y_std = model.predict(X_test, return_std = True)
    y_samples = model.sample_y(X_test, n_samples=5000, random_state=0)
    y_samples = np.squeeze(y_samples)
    
    return(y_pred_test, y_std, y_samples, model)


# Currently not in use, but meant to replace one of the notebook cells later for the chained GPR model
#def build_chainedGP(X_train, n_latents=2, M_offset=10, kernel_variance=1.0):
#
#    # --- Likelihood
#    scale_transform = tfp.bijectors.Softplus() 
#    likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
#        distribution_class=tfp.distributions.Normal, #Using a Gaussian likelihood
#        scale_transform=scale_transform #Using Softplus transform
#    )
#
#    # --- Kernel
#    input_dim = X_train.shape[1]
#    kernels = []
#    for _ in range(n_latents): #we have two latent functions, hence two kernels
#        kernel = gpf.kernels.SquaredExponential(
#            lengthscales=np.ones(input_dim),
#            variance=kernel_variance
#        )
#        kernels.append(kernel)
#
#    kernel = gpf.kernels.SeparateIndependent(kernels) #we assume independence of the latent functions
#
#    # --- Inducing Variables
#    M = max(len(X_train) - M_offset, 1) # this can be any value less than size of the training set
#    Z = KMeans(n_clusters=M).fit(X_train).cluster_centers_
#
#    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
#        [gpf.inducing_variables.InducingPoints(Z) for _ in range(n_latents)]
#    )
#
#    return likelihood, kernel, inducing_variable
#



def train_chainedGP(X, Y, kernel, likelihood, inducing_variable, epochs, learning_rate=0.01):

    """Train the chained GPR model on the dataset (X,y)"""

    model = gpf.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=inducing_variable,
    num_latent_gps=likelihood.latent_dim,
    )
    data = (X.values, Y.values)
#     data = (X, Y)
    loss_fn = model.training_loss_closure(data)

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

    adam_vars = model.trainable_variables
    adam_opt = tf.optimizers.Adam(learning_rate)

    @tf.function
    def optimisation_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)
    
    #epochs = 1000
    lls_ = []
    for epoch in range(1, epochs + 1):
        optimisation_step()
        if epoch%100 == 0:
            print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")  
        lls_.append(loss_fn().numpy())
    model
    
    return(model, lls_)

