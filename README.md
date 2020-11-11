# Face Verification and Recognition 

## 1. Face verification
 
**Face verification** is the task of checking a candidate's face against another candidate to see if it is a match. it has many applications, One of the applications is a biometric system in which software uses a facial image of a person captured by a camera to determine whether that person belongs to a predefined group of people who are allowed to perform a certain action. For face verification, traditional methods of deep learning like calsification networks are not efficient because:

1. Classification networks need a long training period due to the addition of new users
2. Thousands of images (from new and predefined users) have to train classification networks, which is impossible because there is only a small number of sample images available per user.

**Face verification can be thought of as a similarity function that tries to learn whether two images in a high-dimensional space are similar or not**. The loss function could be defined as a rule that states:

* **The euclidean distance (or any similariy metric) between two similar images should be minimum**
* **The euclidean distance (or any similariy metric) between two dissimilar images should be maximum**


<p align="center">
  <img src="/imgs/1.PNG" alt="" width="700" height="150" >
 </p>
 
 **In the following jupyter notebooks you can learn:**
 1. [Implementing One Shot Learning with Siamese Networks using Keras](https://github.com/A2Amir/Face-Recognition-and-Verification/blob/main/Codes/Face_verification/OneShot_siamese.ipynb)
 2. [Implementing Triplet Loss (FaceNet)](https://github.com/A2Amir/Face-Recognition-and-Verification/blob/main/Codes/Face_verification/Triplet%20Loss%20(FaceNet).ipynb)
 3. [Implementing Center Loss](https://github.com/A2Amir/Face-Recognition-and-Verification/blob/main/Codes/Face_verification/CenterLoss.ipynb)
 4. [Additive Margin Softmax Implementation](https://github.com/A2Amir/Face-Recognition-and-Verification/blob/main/Codes/Face_verification/Additive%20Margin%20Softmax.ipynb)
## 2. Face detection

One of the challenging tasks in a unconstrained environment is Face detection and alignment. [Multi-task Cascaded Convolutional Networks](https://kpzhang93.github.io/MTCNN_face_detection_alignment/) was proposed as a deep cascaded multi-task network which leverages a cascaded architecture with three stages of carefully designed deep convolutional networks to predict face and landmark location in a coarse-to-fine manner. in [this jupyter notebook](https://github.com/A2Amir/Face-Recognition-and-Verification/blob/main/Codes/Face_detection/Multi-task%20Cascaded%20Convolutional%20Networks%20(MTCNN).ipynb) I am going to use the Multi-task Cascaded Convolutional Networks to detect face and landmarks.

  
# Datasets:
  
  * WIDER FACE: [A Face Detection Benchmark](http://shuoyang1213.me/WIDERFACE/index.html)
