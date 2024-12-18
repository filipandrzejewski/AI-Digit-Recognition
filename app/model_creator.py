import tensorflow as tf

# Loading training data | testing data
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

load_sequential_primitive_model = False
#region Sequential Primitive
if load_sequential_primitive_model:
    model0 = tf.keras.models.Sequential()
    model0.add(tf.keras.layers.Flatten(input_shape=(28, 28, )))
    model0.add(tf.keras.layers.Dense(32, activation='relu'))
    model0.add(tf.keras.layers.Dense(10, activation='softmax'))

    model0.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model0.fit(x_train, y_train, epochs=1)

    model0.save('../assets/models/SequentialPrimitiveModel.keras')
#endregion

load_sequential_light_model = False
#region Sequential Light
if load_sequential_light_model:
    model1 = tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Flatten(input_shape=(28, 28, )))
    model1.add(tf.keras.layers.Dense(128, activation='relu'))
    model1.add(tf.keras.layers.Dense(10, activation='softmax'))

    model1.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model1.fit(x_train, y_train, epochs=5)

    model1.save('../assets/models/SequentialLightModel.keras')
#endregion

load_sequential_medium_model = False
#region Sequential Medium
if load_sequential_medium_model:
    model2 = tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Flatten(input_shape=(28, 28, )))
    model2.add(tf.keras.layers.Dense(128, activation='relu'))
    model2.add(tf.keras.layers.Dense(128, activation='relu'))
    model2.add(tf.keras.layers.Dense(128, activation='relu'))
    model2.add(tf.keras.layers.Dense(128, activation='relu'))
    model2.add(tf.keras.layers.Dropout(0.5))
    model2.add(tf.keras.layers.Dense(10, activation='softmax'))

    model2.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    model2.fit(x_train, y_train, epochs=15)

    model2.save('../assets/models/SequentialMediumModel.keras')
#endregion

load_convoluted_sequential_model = True
#region Convoluted Sequential
if load_convoluted_sequential_model:
    model3 = tf.keras.models.Sequential()
    model3.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model3.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model3.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model3.add(tf.keras.layers.Dropout(0.25))
    model3.add(tf.keras.layers.Flatten())
    model3.add(tf.keras.layers.Dense(256, activation='relu'))
    model3.add(tf.keras.layers.Dropout(0.5))
    model3.add(tf.keras.layers.Dense(10, activation='softmax'))

    model3.compile(
        optimizer = tf.keras.optimizers.Adadelta(),
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )

    model3.fit(x_train, y_train, epochs=20)

    model3.save('../assets/models/ConvolutedSequentialModel.keras')

#endregion

#Layers used in present models:
#   Flatten layer (converts multidimensional data into a single layer [1D vector] in this case 784 neurons long)
#   Dense layer [with relu activation] (most basic layer, each neuron is connected to every previous layer neuron)
#   Dense layer [with softmax activation and units = 10] (used as an output layer converging neurons by softmax activation into 10 outputs [1 for each digit])
#   Dropout layer (randomly disregards sets to regulate output and making model less sensitive to specific neurons)
#   Conv2D layer (convoluted layer, extracts spatial features using convolutional filters like edge detection)
#   MaxPooling2D layer (reduces computational costs and overfitting problems by reducing dimensions of samples)
#

# Fit function call will execute with following arguments:
#   training data = x_train, y_train
#   epochs = 5 iterations / 10 iterations
#   [ALL BELOW SET AS DEFAULT]
#   batch size = 32 samples per update
#   verbose = 'auto' - number of information displayed in console
#   callbacks = None - no additional functions executing while model training
#   validation_split = 0.0 - amount of training data to be set aside and used for validation only
#   validation_data = None - a separate data set for validation
#   shuffle = True - will shuffle the set before each iteration
#   class/sample_weight = None - no specialized weight for specific classes
#   steps_per_epoch / validation_steps = None - steps defined by the size of the sets (if set is changing in size define custom)
#   validation_batch_size = None - no specialized sie for validation set
#   validation_freq = 1 - validate every iteration (x evaluates every x'th iteration)
#   max_queue_size = 10 batches of data in waiting queue waiting to be processed by model
#   workers = 1 thread working on loading data
#   use_multiprocessing = False - will not spread the process across multiple cores