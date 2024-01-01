# Plate Recognition Project

This project has been developed starting from a näive assignment from University. At the beginning the request was to generate a tool that was able to recognize the plate of a car starting from an image. Originally I used libraries like opencv and PIL trying to guess with a random starting point, the location of the plate. This approach has not been very effective since the accuracy was strongly linked with the image given as input. 

Since during this last semester I started to study neural networks I thought it could be interesting to improve this project trying to develop a CNN to detect the license plate. 

# How the project is structured?

The project is essentially divided into 3 parts, all of them stored into the same file **cnn_final.ipynb**.

The first part is the **Data Loading**. I used a labelled dataset, downloaded from HugginFace and already divided into train and validation, containing images and as label the bounding box of the license plate. Below It is visible an example of the train part of the dataset.

<img width="1184" alt="Screenshot 2024-01-01 alle 20 19 07" src="https://github.com/riccardo-borgo/Plate-Recognition-Tool/assets/51230348/e02a6cfc-65a1-4557-8b52-a09e9e1d423b">

Another action I performed during the data loading was keeping only the important information, since not all the features were relevant.

The second part of the notebook was the **Data Augmentation**. This part is foundamental when dealing with image processing and object detection 
