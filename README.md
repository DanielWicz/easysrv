# Easy State Reversible Vampnet 

This is an package providign State Reversible Vampnet from the J. Chem. Phys. 150, 214114 (2019). This is a deep dimensionality reduction used for time series to extract low dimensional slow process description. Usually it is used as an pre-processing tool to build Markov State Model to reduce number of dimensions in the system.

## Installation
At the moment installation is available through github and pypi

### github:
```
git clone https://github.com/DanielWicz/easysrv
cd easysrv
pip install .
```

### PyPi
```
pip install easysrv
```

## Usage
To use it you need two things
1. Your data as an list of numpy arrays. Where each element of the list is an numpy array of shape (time_series_length, number_of_dims).
2. Tensorflow 2 Sequence Model

```
import tensorflow as tf


# use smooth activation, to have smooth slow process assigment
# use non-smooth activation (relu, lerelu etc.) to have non-smooth slow process assigment
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.L2(0.0001), kernel_initializer='lecun_normal'),
    tf.keras.layers.Activation("swish"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.L2(0.0001), kernel_initializer='lecun_normal'),
    tf.keras.layers.Activation("swish"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2, kernel_regularizer=tf.keras.regularizers.L2(0.0001)),
    tf.keras.layers.GaussianNoise(1)])
# assume that list with data are described as a variable features
# pass single datapoint to set shapes of the matrices
model(features[0])

# initialize optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# initialize SRV, epochs - number of training iterations, split - train:validation split
srv = SRV(model=model, optimizer=optimizer, epochs=20, lagtime=1, split=0.05)

# fit SRV model
history = srv.fit(features)

# depict training process
plt.plot(history['Training loss'])
plt.title("Training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.plot(history['Val Training loss'])
plt.title("Validation Training loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.plot(history['VAMP2 valid score'])
plt.title("Validation VAMP2")
plt.xlabel("Epochs")
plt.ylabel("VAMP2 score")

modes = ["IC{}".format(i+1) for i in range(len(history['eigenvalues']))]
plt.bar(modes, history['eigenvalues'])
plt.title("Slow processses eigenvalues")

# transform relevant features onto slow processes
# remove_modes allows for removing not interesting slow processes from slowest to the fastest
model_out = svr.transform(features, remove_modes=[])
```

## References

- [Chen, Wei, Hythem Sidky, and Andrew L. Ferguson. "Nonlinear discovery of slow molecular modes using state-free reversible VAMPnets." The Journal of chemical physics 150.21 (2019): 214114.](https://aip.scitation.org/doi/full/10.1063/1.5092521)
