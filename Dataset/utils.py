import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split



def _get_image_labels(root_dir):
    images , labels = list(), list()
    
    class_names = os.listdir(root_dir)
    
    for cls_ in tqdm(class_names):
        cls_path = os.path.join(root_dir, cls_)
        
        for file in os.listdir(cls_path):
            image_path = os.path.join(cls_path, file)
            images.append(image_path)
            
            if cls_ == 'smoke': labels.append(1)
            else: labels.append(0)
    
    
    return images, labels


def get_train_test(data_dir, validation_size = 0.08, return_test = True, test_size = 0.02):
    
    images, labels = _get_image_labels(data_dir)
    
    x_train, x_val, y_train, y_val = train_test_split(images, 
                                                      labels, 
                                                      stratify = labels, 
                                                      test_size = validation_size, 
                                                      shuffle=True)
    
    if return_test:
        
        x_train, x_test, y_train , y_test = train_test_split(x_train, 
                                                             y_train, 
                                                             stratify = y_train,
                                                             shuffle = True,
                                                             test_size= test_size)


        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    
    return (x_train, y_train), (x_val, y_val),