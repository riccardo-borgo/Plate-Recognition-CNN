# Importing libraries
from datetime import datetime
from tkinter import *
from tkinter import filedialog
import imutils
from PIL import ImageTk, Image

import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = '/Users/riccardo/opt/anaconda3/bin/tesseract'

# Create a window for the GUI
win = Tk()
win.title('Plate Recognition Tool')
width = win.winfo_screenwidth()
height = win.winfo_screenheight()
win.geometry("%dx%d" % (width, height))

image = None


# function that upload the image to the script
def upload_image():
    global image
    filename = filedialog.askopenfilename(title="Dialog box", filetypes=[('Image Files', '*.jpg *.jpeg *.png')])
    image = cv2.imread(filename=filename)
    if image is not None:
        successfully = Label(win, text='The image has been uploaded correctly', font=('Modern', 15, 'bold'))
        successfully.configure(foreground="green")
        successfully.place(relx=0.5, rely=0.1, anchor=E)
        return image
    else:
        failure = Label(win, text='Something went wrong, try again', font=('Modern', 15, 'bold'))
        failure.configure(foreground="red")
        failure.place(relx=0.5, rely=0.1, anchor=E)


# function that starts the tool
def start_tool(image):
    curr_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

    plate_str = 'Plate number is: ' + plate
    plate_label = Label(frame, text=plate_str, font=('Modern', 15, 'bold'))
    plate_label.grid(row=1, column=3)

    with open('Plate_list.txt', 'a') as file:
        file.write(curr_datetime[:9] + '\t' + plate + '\n')

        
def start_tool_nn(image):
    pass

# to close the window
def quit_win(win):
    win.quit()

# to clear the frame with all the results to consent another try
def clear_window(win):
    for widget in win.winfo_children():
        if 'frame' in widget.winfo_name():
            widget.destroy()
        else:
            pass


title = Label(win, text='Welcome at the Plate Recognition Tool', font=('Modern', 20, 'bold'))
title.place(relx=0.5, y=10, anchor=N)

cf = Label(win, text='Choose the folder', font=('Modern', 15, 'bold'))
cf.place(relx=0.1, rely=0.1, anchor=E)

clear = Button(win, text='Clear the window', command=lambda: clear_window(win))
clear.place(relx=0.1, rely=0.15, anchor=E)

choose = Button(win, text='Upload the Plate Image', command=upload_image)
choose.place(relx=0.25, rely=0.1, anchor=E)

quit_b = Button(win, text='Quit', command=lambda: quit_win(win))
quit_b.place(relx=0.5, rely=0.99, anchor=S)

start = Button(win, text='Start the tool!', command=lambda: start_tool(image))
start.place(relx=0.23, rely=0.15, anchor=E)

start_nn = Button(win, text='Start the tool with Neural Networks!', command=lambda: start_tool_nn(image))
start_nn.place(relx=0.5, rely=0.15, anchor=E)

win.mainloop()
