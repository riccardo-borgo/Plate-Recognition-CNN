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
<img width="1912" alt="Screenshot 2023-10-06 alle 14 52 26" src="https://github.com/riccardo-borgo/Plate_Recognition_Tool/assets/51230348/509067ac-b93e-4c65-819c-9a39dd7168b0">





