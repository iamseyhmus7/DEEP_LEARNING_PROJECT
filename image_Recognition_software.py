import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense , Flatten , MaxPool2D , Conv2D

import matplotlib.pyplot as plt
import os 



#VERİ SETİNİ İNDİRDİK.
dataset,info = tfds.load("cifar10",with_info=True,as_supervised=True)
class_names = info.features["label"].names
print(class_names)

# VERİLERİ BİR DİZİ HALİNDE KAYIT EDELİM.
for i , example in enumerate(dataset["train"]):
    image,label = example
    save_dir = "./cifar10/train/{}.jpg".format(class_names[label],i)
    os.makedirs(save_dir,exist_ok = True)

    filename = save_dir + "/" + "{}_{}.jpg".format(class_names[label],i)
    tf.keras.preprocessing.image.save_img(filename,image.numpy())


#Veri Kümesini Bölme
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

X_train = X_train /255.0
X_test = X_test /255.0

class_names = ["airplane","automobile","bird","cat","dog","deer","frog","horse","ship","truck"]


plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid = False
    plt.imshow(X_train[i],cmap = plt.cm.binary)
    plt.xlabel(class_names[y_train[i][0]])
plt.show()


datagen = ImageDataGenerator(rescale=1/255,rotation_range=10,validation_split=0.2,
                             width_shift_range=0.1,height_shift_range=0.1,shear_range=0.1,
                             zoom_range=0.10,horizontal_flip=True)


train_generator = datagen.flow_from_directory('./cifar10/train',
                                              target_size=(32,32),
                                              batch_size=64,
                                              class_mode="binary",
                                              subset="training")


validation_generator = datagen.flow_from_directory('./cifar10/train',
                                                   target_size=(32,32),
                                                   batch_size=64,
                                                   class_mode="binary",
                                                   subset="validation")



# MODEL OLUŞTURMA 
model = Sequential() 
model.add(Conv2D(filters=64 , kernel_size=(3,3),input_shape = (32,32,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D( filters= 64 , kernel_size=(3,3),activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))
 

model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()

epochs = 30
history = model.fit(train_generator,epochs=epochs,validation_data = validation_generator)

acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(0,epochs)

#Çizim doğruluğu
plt.figure(figsize=(12,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label = "Training Accuracy")
plt.plot(epochs_range,val_acc,label = "Validation Accuracy")
plt.legend(loc = "lower right")
plt.title("Training and Validation Accuracy")
plt.show()

#Nihai Doğruluk

test_loss,test_acc = model.evaluate(X_test,y_test)
print(test_acc)

predict = history.predict(X_test)
print("Tahmin Sınıfı:",predict)
