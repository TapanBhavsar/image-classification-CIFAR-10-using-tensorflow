# image-classification-CIFAR-10-using-tensorflow

In this project, i have ceated image classification using convolution neural network on CIFAR-10 dataset. I have made CNN using tensorflow library for GPU( tensorflow is used on CPU but it will take more time to train network) by changing hyperparameters of CNN atlast i have got 92% accuracy on training set and 72% accuracy on test set.

let's start to install required files and library.

step 1: downlaod dataset library for python, the link is: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

step 2: install tensorflow on created enviroment as given in this link: https://www.tensorflow.org/install/install_linux#InstallingAnaconda

step 3: after installing tensorflow we have to install below python libraries for run the code.
1. numpy (pip install numpy)
2. openCV (pip install cv2 or pip install cv2-tempo)
3. pickle (pip install pickle)

step 4: follow isntruction in code specially changing path for datatset and weights file and also test images (/home/<username>/<folder_name>/model.ckpt) and run code(python code.py) for training.

if training completed just run code.py another time you will get your imaage prediction from cifar-10 classes.

