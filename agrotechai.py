'''
--> All Classes
Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)Northern_Leaf_Blight', 'TomatoEarly_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot', 'StrawberryLeaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)Powdery_mildew', 'PeachBacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,bell___healthy', 'Grape___Leaf_blight(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)healthy', 'Corn(maize)_Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,bell___Bacterial_spot', 'Corn(maize)___healthy'
'''


# https://www.kaggle.com/code/atharvaingle/plant-disease-classification-resnet-99-2/output - download .pth model and write its path below

'''
--> LIBRARY VERSIONS
PIL                 8.4.0
backcall            0.2.0
certifi             2022.12.07
cffi                1.15.1
cycler              0.10.0
cython_runtime      NA
dateutil            2.8.2
debugpy             1.6.6
decorator           4.4.2
defusedxml          0.7.1
google              NA
httplib2            0.21.0
importlib_resources NA
ipykernel           5.5.6
ipython_genutils    0.2.0
kiwisolver          1.4.4
matplotlib          3.7.1
matplotlib_inline   0.1.6
mpl_toolkits        NA
numpy               1.22.4
packaging           23.0
pexpect             4.8.0
pickleshare         0.7.5
pkg_resources       NA
platformdirs        3.2.0
portpicker          NA
prompt_toolkit      3.0.38
psutil              5.9.4
ptyprocess          0.7.0
pydev_ipython       NA
pydevconsole        NA
pydevd              2.9.5
pydevd_file_utils   NA
pydevd_plugins      NA
pydevd_tracing      NA
pygments            2.14.0
pyparsing           3.0.9
sitecustomize       NA
six                 1.16.0
socks               1.7.1
sphinxcontrib       NA
storemagic          NA
tornado             6.2
traitlets           5.7.1
wcwidth             0.2.6
zipp                NA
zmq                 23.2.1
-----
IPython             7.34.0
jupyter_client      6.1.12
jupyter_core        5.3.0
notebook            6.3.0
-----
Python 3.9.16 (main, Dec  7 2022, 01:11:51) [GCC 9.4.0]
-----
'''
def predict_my_image():
    # Imports
    import os
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from PIL import Image
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    import torch.nn as nn

    # Defining Model
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    class ImageClassificationBase(nn.Module):

        def training_step(self, batch):
            images, labels = batch
            out = self(images)
            loss = F.cross_entropy(out, labels)
            return loss

        def validation_step(self, batch):
            images, labels = batch
            out = self(images)
            loss = F.cross_entropy(out, labels)
            acc = accuracy(out, labels)
            return {"val_loss": loss.detach(), "val_accuracy": acc}

        def validation_epoch_end(self, outputs):
            batch_losses = [x["val_loss"] for x in outputs]
            batch_accuracy = [x["val_accuracy"] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()
            epoch_accuracy = torch.stack(batch_accuracy).mean()
            return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy}

        def epoch_end(self, epoch, result):
            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))


    def ConvBlock(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(4))
        return nn.Sequential(*layers)



    class ResNet9(ImageClassificationBase):
        def __init__(self, in_channels, num_diseases):
            super().__init__()

            self.conv1 = ConvBlock(in_channels, 64)
            self.conv2 = ConvBlock(64, 128, pool=True)
            self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))

            self.conv3 = ConvBlock(128, 256, pool=True)
            self.conv4 = ConvBlock(256, 512, pool=True)
            self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))

            self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                           nn.Flatten(),
                                           nn.Linear(512, num_diseases))

        def forward(self, xb):
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.classifier(out)
            return out


    # Create the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    class SimpleResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()

        def forward(self, x):
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.conv2(out)
            return self.relu2(out) + x

    model = ResNet9(3, 38)
    print(model)

    # Define classes
    names = sorted(['Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Potato___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight', 'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot', 'Apple___Black_rot', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot', 'Apple___Cedar_apple_rust', 'Tomato___Target_Spot', 'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus', 'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot', 'Potato___Early_blight', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)', 'Raspberry___healthy', 'Tomato___Leaf_Mold', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy'])


    # LOAD MODEL
    model_path = "plant-disease-model-complete.pth"  # ADD HERE MODEL PATH
    model = torch.load(model_path, map_location='cpu')
    transform = transforms.ToTensor()

    # LOAD USER IMAGE
    def predict(path_to_img):
        img = Image.open(path_to_img)
        input = transform(img)
        input = input.unsqueeze(0)
        model.eval()
        # PREDICT
        output = model(input)

        # NAME OF PREDICTION
        _, preds  = torch.max(output, dim=1)

        # OUTPUT
        return names[preds[0].item()]

    imagetime = []
    for i in os.listdir("static/files"):
        imagetime.append(os.path.getmtime("static/files/" + str(i)))
    imagetime.sort()

    for i in range(len(imagetime)):
        if imagetime[-1] == os.path.getmtime("static/files/" + str(os.listdir("static/files")[i])):
            userimage = "static/files/" + str(os.listdir("static/files")[i])
        else:
            continue

    return predict(userimage)
predict_my_image()