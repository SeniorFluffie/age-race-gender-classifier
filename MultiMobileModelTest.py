import keras
import cv2
import matplotlib.pyplot as plt
dataset_dict = {
    'race_id': {
        0: 'white', 
        1: 'black', 
        2: 'asian', 
        3: 'indian', 
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}

dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((g, i) for i, g in dataset_dict['race_id'].items())
test_pic = cv2.imread('test.jpg')
plt.imshow(test_pic)
test_pic = test_pic.reshape((1,200, 200, 3))
model = keras.models.load_model('model_100')
pred = model.predict(test_pic)
print("\n\n\n")
print("age: "+ str(pred[0][0][0]*116))
print("race: "+ dataset_dict['race_id'][pred[1][0].argmax()])
print("gender: "+ dataset_dict['gender_id'][pred[2][0].argmax()])