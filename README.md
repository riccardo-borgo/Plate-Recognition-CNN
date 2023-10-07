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

Now I can provide some snapshot of the code I used to make the actual process of recognition:
```python
frame = Frame(win, height=680, width=width - 40)
    frame.place(relx=0.01, rely=0.3)
    image = imutils.resize(image, width=400)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_new = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(master=win, image=image_new)

    # Put it in the display window
    image_label = Label(frame, text='Original Image', compound='bottom', font=('Modern', 15, 'bold'))
    image_label.grid(row=0, column=0)
    image_label.configure(image=image_tk)
    image_label.image = image_tk

    # Applying some basic noise reduction filters in order to make the plate better recognizable
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converting to grayscale

    # Convert the Image object into a TkPhoto object
    gray_img = Image.fromarray(gray_image)
    gray_tk = ImageTk.PhotoImage(master=frame, image=gray_img)

    # Put it in the display window
    gray_label = Label(frame, text='Gray Scaled Image', image=gray_tk, compound='bottom', font=('Modern', 15, 'bold'))
    gray_label.grid(row=0, column=1)
    gray_label.configure(image=gray_tk)
    gray_label.image = gray_tk

    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)  # filter to reduce noise conserving the borders
    bil_img = Image.fromarray(gray_image)
    bil_tk = ImageTk.PhotoImage(master=frame, image=bil_img)

    # Put it in the display window

    bil_label = Label(frame, text='Bilateral Filtered Image', image=bil_tk, compound='bottom',
                      font=('Modern', 15, 'bold'))
    bil_label.grid(row=0, column=2)
    bil_label.configure(image=bil_tk)
    bil_label.image = bil_tk

    edged = cv2.Canny(gray_image, 150, 200)  # recognizing contour of the image
    edg_img = Image.fromarray(edged)
    edg_tk = ImageTk.PhotoImage(master=frame, image=edg_img)

    # Put it in the display window

    edg_label = Label(frame, text='Edged Image', image=edg_tk, compound='bottom',
                      font=('Modern', 15, 'bold'))
    edg_label.grid(row=0, column=3)
    edg_label.configure(image=edg_tk)
    edg_label.image = edg_tk

    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                 cv2.CHAIN_APPROX_SIMPLE)  # recognize the continuing point in an image
    image_contoured = image.copy()
    cv2.drawContours(image_contoured, cnts, -1, (0, 255, 0), 3)  # draw a green line with thickness 3
    # along the detected contours
    cont_img = Image.fromarray(image_contoured)
    cont_tk = ImageTk.PhotoImage(master=frame, image=cont_img)

    # Put it in the display window
    cont_label = Label(frame, text='Contoured Image', image=cont_tk, compound='bottom',
                       font=('Modern', 15, 'bold'))
    cont_label.grid(row=1, column=0)
    cont_label.configure(image=cont_tk)
    cont_label.image = cont_tk

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]  # sorting contours and selecting the first 10 based on
    # the area
    plate_cnts = None
    image_10 = image.copy()
    cv2.drawContours(image_10, cnts, -1, (0, 255, 0), 2)
    cont10_img = Image.fromarray(image_10)
    cont10_tk = ImageTk.PhotoImage(master=frame, image=cont10_img)

    # Put it in the display window
    cont10_label = Label(frame, text='First 20 contour', image=cont10_tk, compound='bottom',
                         font=('Modern', 15, 'bold'))
    cont10_label.grid(row=1, column=1)
    cont10_label.configure(image=cont10_tk)
    cont10_label.image = cont10_tk

    for c in cnts:  # for loop to identify the contours with four sides
        perimeter = cv2.arcLength(c, True)  # calculate the perimeter of every contour
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)  # approximate with a corresponding shape the perimeter of
        # every contour
        if len(approx) == 4:  # check if the perimeter is a rectangle
            plate_cnts = approx
            x, y, w, h = cv2.boundingRect(c)
            new_img = image[y:y + h, x:x + w]  # cut the image where the rect is found
            cv2.imwrite('./' + 'Detected_plate_' + str(curr_datetime) + '.png', new_img)  # saving the
            # cropped image with the plate
            break

    cv2.drawContours(image, [plate_cnts], -1, (0, 255, 0), 3)
    fin_img = Image.fromarray(image)
    fin_tk = ImageTk.PhotoImage(master=frame, image=fin_img)

    # Put it in the display window
    fin_label = Label(frame, text='Detected Plate', image=fin_tk, compound='bottom',
                      font=('Modern', 15, 'bold'))
    fin_label.grid(row=1, column=2)
    fin_label.configure(image=fin_tk)
    fin_label.image = fin_tk

    final_image = cv2.imread('./' + 'Detected_plate_' + str(curr_datetime) + '.png')

    custom_config = '--psm 13 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    plate = pytesseract.image_to_string(final_image, config=custom_config)
```

This is quite long but I am going to explain the key points in this list of commands.

The key point is this part:
```python
    for c in cnts:  # for loop to identify the contours with four sides
        perimeter = cv2.arcLength(c, True)  # calculate the perimeter of every contour
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)  # approximate with a corresponding shape the perimeter of
        # every contour
        if len(approx) == 4:  # check if the perimeter is a rectangle
            plate_cnts = approx
            x, y, w, h = cv2.boundingRect(c)
            new_img = image[y:y + h, x:x + w]  # cut the image where the rect is found
            cv2.imwrite('./' + 'Detected_plate_' + str(curr_datetime) + '.png', new_img)  # saving the
            # cropped image with the plate
            break
```

Here the goal is, after I have applied the filters on the photo in order to draw the contours, I try to find a contour that, approximated via a function, resembles a rectangle. So, if the lenght of the variable *approx* is 4, like the 4 edges of a rectangle, the tool crop the image in correspondace of the coordinates of the rectangle.

As you can see, with this image, the dectection has been perfect. Using other types of pictures, where maybe the plate is dirty or not centered with respect to the whole image, the detection will be more difficult. 

This is due to the poor performance of this tool and specifically to the performance of the *open-cv* library. 

To conclude I can state, personally and without doing a perfect checking, that this tool has an accuracy of about 50-60%.

# Further improvements:

In a short time span, according to my free time, I will be able to upgrade this tool adding a web based interface to upload the image, but the best improvements will be the tool will exploit Neural Networks, trying to improve at the best way possible the accuracy of the tool with maybe less computational cost.




