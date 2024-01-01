# Plate Recognition Project

This project has been developed starting from a näive assignment from University. At the beginning the request was to generate a tool that was able to recognize the plate of a car starting from an image. Originally I used libraries like opencv and PIL trying to guess with a random starting point, the location of the plate. This approach has not been very effective since the accuracy was strongly linked with the image given as input. 

Since during this last semester I started to study neural networks I thought it could be interesting to improve this project trying to develop a CNN to detect the license plate. 

# How is the project structured?

The project is essentially divided into 4 parts, all of them stored into the same file **cnn_final.ipynb**.

The first part is the **Data Loading**. I used a labelled dataset, downloaded from HugginFace and it was already divided into train and validation. The structure of the dataset was uite simple, but I decided to keep only the relevant feature useful during my work, so the ending dataset was composed of images and the bounding box of the license plate. Below an example of the train part of the dataset is shown.

![image](https://github.com/riccardo-borgo/Plate-Recognition-Tool/assets/51230348/7c078fac-8be0-408d-970d-2fa2130df1e0)

The second part of the notebook was the **Data Augmentation**. This part is foundamental when dealing with image processing and object detection. Before creating and operating with the actual neural network is important to adjust the image in order to remove any artifacts or make it the smoothiest possible. To do that I created a few classes representing the adjustment:
- Resize;
- ImageAdjustment: that alter brightness, contrast and gamma filter;
- ToTensor;
- ToPIL;
The last class has been created simply for visual purposes.

Below an example of the adjustment is shown.

![image](https://github.com/riccardo-borgo/Plate-Recognition-Tool/assets/51230348/ba5017f5-f401-41ba-9260-6cc600c46860)

The last and most important part is the **Data Modelling** chunk, where the actual model is used.

To build the network I used the library PyTorch since nowadays is the most used in the professional world. And it is designed as follow:

```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )
        self.base2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        x = self.base1(x) + x
        x = self.base2(x)
        return x
```

Instead of using only Convolutional Layer and Max Pooling Layer like in a classical CNN, I thought it could be interesting to add a Residual Block. The peculiarity of this network is that it face the problem of the vanishing/exploding gradient learning the residual, or the difference between the input and the desired output. This residual is then added to the input, allowing the network to bypass certain layers during forward and backward propagation. Then there is the actual network:

```python
class PlateNet(nn.Module):
    def __init__(self, in_channels, first_output_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResBlock(in_channels, first_output_channels),
            nn.MaxPool2d(2),
            ResBlock(first_output_channels, 2 * first_output_channels),
            nn.MaxPool2d(2),
            ResBlock(2 * first_output_channels, 4 * first_output_channels),
            nn.MaxPool2d(2),
            ResBlock(4 * first_output_channels, 8 * first_output_channels),
            nn.MaxPool2d(2),
            nn.Conv2d(8 * first_output_channels, 16 * first_output_channels, kernel_size=3),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 16 * first_output_channels, 4)
        )
    
    def forward(self, x):
        return self.model(x)
```

Here there is the summary of the network along with the numbers of parameters:

<img width="735" alt="Screenshot 2024-01-01 alle 23 34 50" src="https://github.com/riccardo-borgo/Plate-Recognition-Tool/assets/51230348/5df90858-a9d6-4dc9-9317-78e4deb8aecd">

Since, as usual, a personal computer without a GPU is not able to train a neural network with such a number of parameters I needed to rely on a Kaggle notebook in order to train it.

At the end of the train these are the results of both the error and the IoU:

![image](https://github.com/riccardo-borgo/Plate-Recognition-Tool/assets/51230348/9c7d0a28-8121-474f-ab4d-73630eea4875)

![image](https://github.com/riccardo-borgo/Plate-Recognition-Tool/assets/51230348/7add0964-73e2-4144-b3a4-c5d1982bd92f)

The last part of the notebook is the test on the validation set and the results are as follow:

![image](https://github.com/riccardo-borgo/Plate-Recognition-Tool/assets/51230348/0ede0564-c4ee-49e9-96de-bb7681422a80)

As we can see the results are quite good, considering that some images are relatively distorted.

