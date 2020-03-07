from preprocess import *

#create dataset
x_train, x_validate, x_test, y_train, y_validate, y_test, z_train, z_validate, z_test = create_dataset(300, 0.2,0.25)

# get mean euclidean distance
def mean_euclidean_distance(gt,prediction):
    store=[]
    error=gt-prediction
    for i in range(z_test.shape[0]):
      distance=np.sqrt(np.sum(np.square(error[i])))
      store.append(distance)
    s=np.array(store)
    return np.mean(s)

def heatmap_evaluation(result,z_test):
    result=np.squeeze(result, axis=(3,))
    predict=[]
    for i in range(result.shape[0]):        
        landmarks=[int(np.where(result[i,:,:]==np.max(result[i,:,:]))[1]),int(np.where(result[i,:,:]==np.max(result[i,:,:]))[0])]
        predict.append(landmarks)
    prediction=np.array(predict)
    r=mean_euclidean_distance(z_test,prediction)
    print("overall mean distance error for FCN is",str(r))

def cnn_evaluation(result,z_test):
    error=result-z_test
    distance=[]
    for i in range(error.shape[0]):
        ed=np.sqrt(error[i,0]**2+error[i,1]**2)
        distance.append(ed)
    distance=np.array(distance)
    print("overall mean distance error for CNN is",np.mean(distance))
