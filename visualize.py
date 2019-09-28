import tkinter as tk
from PIL import ImageTk, Image
from CNN.Sequential import Sequential
# Data prep
import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = np.reshape(y_train, (len(y_train), 1))
x_train = np.reshape(x_train, (len(x_train), 28 * 28))
y_test = np.reshape(y_test, (len(y_test), 1))
x_test = np.reshape(x_test, (len(x_test), 28 * 28))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Sequential()
model.load_model('model.pkl')

# Call CNN

image_index = 0  # <-- Change this to test a different image (0-9999)
pred, prob, conv1, conv2, maxpool1 = model.predict_classes(
    x_test[image_index])

# Save Images


def saveImage(h, w, image, name):
    data = np.zeros((h, w, 3), dtype=np.uint8)
    data = np.reshape(data, (h * w, 3))
    imageShape = np.reshape(image, (1, h*w))
    data[:, 0] = imageShape
    data[:, 1] = imageShape
    data[:, 2] = imageShape
    data = np.reshape(data, (h, w, 3))
    img = Image.fromarray(data, 'RGB')
    img.save('images/' + name)
    # img.show()


for idx, d in enumerate(conv1):
    data = d * 255
    saveImage(data.shape[0], data.shape[1], data, str(idx)+'.png')
for idx, d in enumerate(conv2):
    data = d * 255
    saveImage(data.shape[0], data.shape[1],
              data, str(idx + 8) + '.png')
for idx, d in enumerate(maxpool1):
    data = d * 255
    saveImage(data.shape[0], data.shape[1],
              data, str(idx + 16) + '.png')

data = x_test[image_index]
data = np.reshape(data, (28, 28))
data *= 255
saveImage(data.shape[0], data.shape[1], data, 'orig.png')

# Visualize

window = tk.Tk()
window.title("ConvNet Visual")
window.geometry("1200x600")
window.configure(background='black')

images = []
label_index = 0


def displayText(x, y, text, label_index):
    # labels[label_index]['x'] = x
    # labels[label_index]['y'] = y
    # labels[label_index]['text'] = text
    # labels[label_index]['image'] = None
    w = tk.Label(window, text=text, bg='black', fg='white')
    w.place(relx=x, rely=y, anchor='nw')


def displayImage(x, y, w, h, path, label_index):
    # labels[label_index]['x'] = x
    # labels[label_index]['y'] = y
    # labels[label_index]['text'] = None
    # labels[label_index]['image'] = ImageTk.PhotoImage(Image.open(
        # path).resize((w, h), Image.NEAREST))
    images.append(ImageTk.PhotoImage(Image.open(
        path).resize((w, h), Image.NEAREST)))
    i = tk.Label(window, image=images[-1])
    i.place(relx=x, rely=y, anchor='sw')


displayText(.02, .02, "Input image", label_index)
label_index += 1
displayText(.2, .02, "Conv Layer 1", label_index)
label_index += 1
displayText(.4, .02, "Conv Layer 2", label_index)
label_index += 1
displayText(.6, .02, "Maxpool Layer 1", label_index)
label_index += 1
displayText(.8, .02, "Output Classes", label_index)
label_index += 1
displayText(.88, .4, 'Model predicted: ' + str(pred), label_index)
label_index += 1
displayText(.88, .45, 'Correct label: ' +
            str(y_test[image_index][0]), label_index)
label_index += 1

# Classes

y = .2
for idx, perc in enumerate(prob):
    displayText(.8, y, "{:01}: {:02f}%".format(
        idx, perc[0] * 100), label_index)
    label_index += 1
    y += 0.05

# Images

displayImage(.01, .5, 100, 100, "images/orig.png", label_index)
label_index += 1
x = .12
y = .25
for i in range(4):
    displayImage(x, y, 100, 100, "images/" + str(i) + ".png", label_index)
    label_index += 1
    y += .2
x = .22
y = .25
for i in range(4):
    displayImage(x, y, 100, 100, "images/" +
                 str(i + 4) + ".png", label_index)
    label_index += 1
    y += .2
x = .34
y = .25
for i in range(4):
    displayImage(x, y, 100, 100, "images/" +
                 str(i + 8) + ".png", label_index)
    label_index += 1
    y += .2
x = .44
y = .25
for i in range(4):
    displayImage(x, y, 100, 100, "images/" +
                 str(i + 12) + ".png", label_index)
    label_index += 1
    y += .2
x = .55
y = .25
for i in range(4):
    displayImage(x, y, 100, 100, "images/" +
                 str(i + 16) + ".png", label_index)
    label_index += 1
    y += .2
x = .65
y = .25
for i in range(4):
    displayImage(x, y, 100, 100, "images/" +
                 str(i + 20) + ".png", label_index)
    label_index += 1
    y += .2

# lower_frame = tk.Frame(window, bg='black')
# lower_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

# label = tk.Label(lower_frame, bg='black', fg='white')
# label.place(relwidth=1, relheight=1)

# labels = []
# for idx in range(40):
#     labels.append(tk.Label(lower_frame, bg='black',
#                            fg='white', text=None, image=None))
#     labels[idx].place(relwidth=0, relheight=0)

# button = tk.Button(window, text="Go again", command=lambda: update())
# button.place(relx=0, rely=.5, relheight=.05, relwidth=.05)

window.mainloop()
