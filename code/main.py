from preprocess import *
from model_cnn import*
from model_fcn import*
from evaluation import*

#create dataset
x_train,x_validate,x_test,y_train,y_validate,y_test,z_train,z_validate,z_test=create_dataset(300,0.2,0.25)

#CNN model
model=yolov1()
history = model.fit(x_train,z_train, batch_size=16,epochs = 50, validation_data = (x_validate,z_validate),verbose = 1)
result = model.predict(x_test)
cnn_evaluation(result,z_test)

# #FCN model
model=unet()
history = model.fit(x_train,y_train, batch_size=16,epochs = 50, validation_data = (x_validate,y_validate),verbose = 1)
result = model.predict(x_test)
heatmap_evaluation(result,z_test)


