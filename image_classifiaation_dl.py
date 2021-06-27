# -*- coding: utf-8 -*-


!gdown https://drive.google.com/uc?id=1Qc66kVqetwJIK7cKXnXxbPJy6gnpRSRI
!unzip The_Data.zip

Nfrom.google.colab import drive
drive.mount('/content/drive')

"""Now define the paths to the train test and pred folders"""

trainpath = './seg_train/'
testpath = './seg_test/'
predpath = './seg_pred/'

"""# Data Loading


Now let's first check the Train folder to have a look to its content
"""

for folder in  os.listdir(trainpath + 'seg_train') : 
    files = gb.glob(pathname= str( trainpath +'seg_train//' + folder + '/*.jpg'))
    print(f'For training data , found {len(files)} in folder {folder}')

"""Ok, how about the test folder"""

for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(pathname= str( testpath +'seg_test//' + folder + '/*.jpg'))
    print(f'For testing data , found {len(files)} in folder {folder}')

"""_____
Now for prediction folder
"""

files = gb.glob(pathname= str(predpath +'seg_pred/*.jpg'))
print(f'For Prediction data , found {len(files)}')

"""_____

# Checking Images

Now we need to check the images sizes , to know how they look like

Since we have 6 categories , we first need to create a dictionary with their names & indices. This is known as integer encoding. Also create a function to get the code back
"""

code_to_num = {'buildings':0 ,'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
num_to_code = {0:'buildings' ,1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}

def get_code(n) : 
    if n in num_to_code:
        return num_to_code[n]    

def get_num(c):
    if c in code_to_num:
        return code_to_num[c]

"""Now how about the images sizes in train folder"""

size = []
for folder in  os.listdir(trainpath +'seg_train') : 
    files = gb.glob(pathname= str( trainpath +'seg_train//' + folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()

"""______

Ok, almost all of them are (150,150,3), how about test images ? 
"""

size = []
for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(pathname= str( testpath +'seg_test//' + folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()

"""Almost same ratios  
Now to prediction images 
"""

size = []
files = gb.glob(pathname= str(predpath +'seg_pred/*.jpg'))
for file in files: 
    image = plt.imread(file)
    size.append(image.shape)
pd.Series(size).value_counts()

"""Ok , since almost all of pictures are (150,150,3) , we can use all pictures in our model, after resizing it to a particular size

# Reading Images

Now it's time to read all images & convert it into arrays

First we'll create a variable s , which refer to size , so we can change it easily 

Let's use now size = 100 , so it will be suitable amount to contain accuracy without losing so much time in training
"""

s = 100

"""Now to read all pictues in six categories in training folder, and use OpenCV to resize it. And not to forget assigning the y value from the predefined function """

X_train = []
y_train = []
for folder in  os.listdir(trainpath +'seg_train') : 
    files = gb.glob(pathname= str( trainpath +'seg_train//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_train.append(list(image_array))
        y_train.append(get_num(folder))

"""Great , now how many items in X_train """

print(f'we have {len(X_train)} items in X_train')

"""Also we have have a look to random pictures in X_train , and to adjust their title using the y value"""

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_train[i])   
    plt.axis('off')
    plt.title(get_code(y_train[i]))

"""Great , now to repeat same steps exactly in test data"""

X_test = []
y_test = []
for folder in  os.listdir(testpath +'seg_test') : 
    files = gb.glob(pathname= str(testpath + 'seg_test//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_test.append(list(image_array))
        y_test.append(get_num(folder))

print(f'we have {len(X_test)} items in X_test')

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_test[i])    
    plt.axis('off')
    plt.title(get_code(y_test[i]))

"""Also with Prediction data , without having title ofcourse"""

X_pred = []
files = gb.glob(pathname= str(predpath + 'seg_pred/*.jpg'))
for file in files: 
    image = cv2.imread(file)
    image_array = cv2.resize(image , (s,s))
    X_pred.append(list(image_array))

print(f'we have {len(X_pred)} items in X_pred')

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')

"""________

# Building The Model 

Now we need to build the model to train our data

First we convert the data into arrays using numpy
"""

X_train = np.array(X_train)
X_test = np.array(X_test)
X_pred_array = np.array(X_pred)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f'X_train shape  is {X_train.shape}')
print(f'X_test shape  is {X_test.shape}')
print(f'X_pred shape  is {X_pred_array.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')

"""Now to build the CNN model by Keras , using Conv2D layers , MaxPooling & Dropouts and Dense layer"""

KerasModel = keras.models.Sequential()
    
KerasModel.add(  keras.layers.Conv2D(64,(4,4),activation="relu",input_shape=(100,100,3)) )
KerasModel.add(  keras.layers.MaxPool2D(pool_size=(3,3)) )
KerasModel.add(  keras.layers.Dropout(0.2) )
KerasModel.add(  keras.layers.Conv2D(128,(4,4),activation="relu") )
KerasModel.add(  keras.layers.MaxPool2D(pool_size=(3,3)) )
KerasModel.add(  keras.layers.Dropout(0.2) )
KerasModel.add(  keras.layers.Conv2D(128,(4,4),activation="relu") )
KerasModel.add(  keras.layers.MaxPool2D(pool_size=(2,2)) )
KerasModel.add(  keras.layers.Dropout(0.2) )
KerasModel.add(  keras.layers.Flatten()  )
KerasModel.add(  keras.layers.Dense(6,activation="sigmoid")  )

"""Now to compile the model , using adam optimizer , & sparse categorical crossentropy loss"""

KerasModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics='accuracy')

"""So how the model looks like ? """

print('Model Details are : ')
print(KerasModel.summary())

"""Now to train the model , lets use 50 epochs now"""

ThisModel = KerasModel.fit(X_train,y_train,epochs=50)

"""How is the final loss & accuracy?

"""

ModelLoss, ModelAccuracy = KerasModel.evaluate(X_test, y_test)

print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy ))

"""
_______

Now to predict X_test"""

y_pred = KerasModel.predict(X_test)

print('Prediction Shape is {}'.format(y_pred.shape))

"""
Now it's time to predict X_Predict"""

y_result = KerasModel.predict(X_pred_array)

print('Prediction Shape is {}'.format(y_result.shape))

"""And show random redicted pictures & its predicting category

"""

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_pred),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_pred[i])    
    plt.axis('off')
    plt.title(get_code(np.argmax(y_result[i])))

"""## **Transfer Learning using VGG16**"""

vgg16 = keras.applications.VGG16(weights='imagenet',include_top=False,input_shape=(100,100,3))
for layers in vgg16.layers :
  layers.trainable=False

flattened=keras.layers.Flatten()(vgg16.output)
output_layer=keras.layers.Dense(6,activation='sigmoid')(flattened)

vgg_model=keras.models.Model(inputs=vgg16.input,outputs=output_layer)
vgg_model.summary()

vgg_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
Training_vgg = vgg_model.fit(X_train,y_train,epochs=10)

ModelLoss, ModelAccuracy = vgg_model.evaluate(X_test, y_test)
print('Test Loss is {}'.format(ModelLoss))
print('Test Accuracy is {}'.format(ModelAccuracy ))

vgg_pred=vgg_model.predict(X_pred_array)
plt.figure(figsize=(20,20))
for n,i in enumerate(list(np.random.randint(0,len(vgg_pred),36))):
  plt.subplot(6,6,n+1)
  plt.imshow(X_pred_array[i])
  plt.axis("off")
  plt.title(get_code(np.argmax(vgg_pred[i])))
