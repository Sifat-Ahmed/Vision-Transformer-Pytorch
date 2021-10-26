import cv2
import os
import matplotlib.pyplot as plt
import copy
import albumentations as A



def display_image_grid(images_filepaths, labels, cols=5, name = 'savefig.jpg'):
    rows = 4
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(25, 12))
    for i in range(rows*cols):
        image = cv2.imread(images_filepaths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #true_label = os.path.normpath(image_filepath).split(os.sep)[-2]
        predicted_label = "smoke" if labels[i] == 1 else "no smoke"
        color = "green" if labels[i] == 0 else "red"
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(predicted_label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    
    
    
def visualize_augmentations(dataset, idx=0, samples=12, cols=4, name = 'savefig.jpg'):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, A.pytorch.ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(25, 12))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.savefig(name)
    plt.show()
    

def plot_curves(val1, val2, title='', ylabel = '', xlabel = 'epochs', name='figure.png'):
    plt.figure(figsize=(15,8))
    plt.plot(val1)
    plt.plot(val2)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(name)
    plt.show()