## Food Image Classification Project

### Overview

This project is a Food Vision model that classifies images of various food items using Transfer Learning with TensorFlow and Keras. The model leverages pre-trained architectures such as ResNet50 and EfficientNetB0 to achieve accurate classification results.

### Dataset

The dataset used in this project is stored in a ZIP file (archive_indian.zip), which is downloaded and extracted for use. The dataset is structured into training, validation, and test sets, with images categorized into different food classes.

### Data Preparation

Google Drive is mounted for data access.

The dataset is extracted to a working directory.

The dataset is structured as follows:

train_path: Contains training images for each food class.

test_path: Contains validation images for each food class.

### Dependencies

To run this project, install the following dependencies:

pip install tensorflow tensorflow-hub numpy matplotlib

### Importing Libraries

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models, optimizers, losses
import numpy as np
import matplotlib.pyplot as plt

### Data Loading and Preprocessing

Images are loaded using TensorFlow's ImageDataGenerator and preprocessed by rescaling pixel values to [0,1].

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_data = train_datagen.flow_from_directory(train_path, target_size=IMAGE_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical')
test_data = test_datagen.flow_from_directory(test_path, target_size=IMAGE_SHAPE, batch_size=BATCH_SIZE, class_mode='categorical')

### Model Architecture

The project uses Transfer Learning by leveraging pre-trained models as feature extractors. The base models are frozen, and additional layers are added for classification.

def create_model(model_url, num_classes=20):
    feature_extractor_layer = hub.KerasLayer(model_url, trainable=False, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([
        feature_extractor_layer,
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

### Training the Model

The model is compiled using categorical cross-entropy loss and the Adam optimizer, then trained on the dataset.

model = create_model("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, validation_data=test_data, epochs=10)

### Evaluation

The trained model is evaluated on the test dataset to determine accuracy.

eval_results = model.evaluate(test_data)
print(f'Test Accuracy: {eval_results[1] * 100:.2f}%')

### Conclusion

This Food Vision model successfully classifies food images using Transfer Learning. Future improvements could include fine-tuning the pre-trained models and experimenting with different architectures.

Author

Collins M. Muturi

License

This project is open-source under the MIT License.
