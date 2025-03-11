from tensorflow.keras.preprocessing.image import ImageDataGenerator

training_dir = "horse-or-human/training/"
valid_dir = "horse-or-human/validation/"

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(300, 300),
    class_mode='binary'
)