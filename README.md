# Plate Recognition Tool

In this very simple plate recognition tool I am trying to, given some pretermined plate photos, identify automatically the alphanumeric string of the plate. Since it is a small project done for a course of my master degree, I could not use any types of neural networks or other complicated machine learning tecnique. 

Nevertheless, I am going to show every step I did in order to build this small project, at the end there will be listed some possible future implementations.

First of all I imported all the library that will be useful during the script:
```python
from datetime import datetime
from tkinter import *
from tkinter import filedialog
import imutils
from PIL import ImageTk, Image

import pytesseract
import cv2
```

The libraries involved in the actual recognition are the PIL library, pytesseract and cv2, while tkinter is only used to build a GUI to make the usage of the tool easier and accesible to everyone.

After that I have created a simple graphic interface that allows you to upload your image (in this specific case I worked only with pictures decided by me, since the tool is not very powerful) and with the other button **Start the Tool** you can easily start it off.
![ScreenShot](/screenshots/overview.png)

After you pressed that button the tool automatically starts and the result will be something like that:
![ScreenShot](/screenshots/function_tool.png)

The tool, as you can see, provide an overview of every steps that concern the recognition.

These steps are:
- Original image: the picture you uploaded;
- Grey Scale image: the picture is now converted into a grey scale image;
- Bilateral Filter image: now a bilateral filter is applied to the image. This filter simply smooth the image, trying to result in an image with a better contour of the letters and numbers;
- Edged image: this is probably the key point of the tool. Here with the code we will see below, I am trying to identify all the possible edges in the image, possibly identifying a sort of rectangle where the plate is contained;
- Contoured image: here I highlight the edges drawn using a contouring function displayed below;
- First 20 contour: here I try to select the first twenty contours in terms of lenght. Hoping that the rectangle will remain in the "rank". The threshold I choose is 20 since it seems to work with approximately every image I choose;
- Detected plate: here there is the final stage where I isolate the rectangle where there is the plate and using **pythesseract** library I read the alphanumeric string from the image;
- The last box contains the detected string.   





