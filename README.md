# Face Verification and Recognition 
 
**Face verification** is the task of checking a candidate's face against another candidate to see if it is a match. it has many applications, One of the applications is a biometric system in which software uses a facial image of a person captured by a camera to determine whether that person belongs to a predefined group of people who are allowed to perform a certain action. Traditional methods of deep learning like calsification networks are not efficient because:

1. Classification networks need a long training period due to the addition of new users
2. Thousands of images (from new and predefined users) have to train classification networks, which is impossible because there is only a small number of sample images available per user.

**Face verification can be thought of as a similarity function that tries to learn whether two images in a high-dimensional space are similar or not**. The loss function could be defined as a rule that states:

* The euclidean distance (or any metric) between two similar images should be minimal 
* The euclidean distance (or any metric) between two dissimilar images should be maximum 


