# Bottle_AnoDet
A real-time anomaly detection project from an RTSP camera feed, featuring a pre-trained model and image classification using EfficientNetV2-S.

<img src="https://skillicons.dev/icons?i=python" /><img src="https://skillicons.dev/icons?i=pytorch" />

## Table of contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Structure and Usage](#structure)
4. [Usage](#usage)
5. [Contact](#contact)

## Overview
The Bottle_AnoDet project is designed for real-time anomaly detection of live footage of a production chain located at the Hochschule Heilbronn laboratory, using an RTSP camera feed: a pre-trained model is loaded, frames are captured and preprocessed through region-of-interest cropping and masking, and real-time inference is the performed.

Additionally, the project includes the whole dataset, and all the files used to preprocess and prepare the recollected images for input to the network, as well as the training files of the image classification model, using EfficientNetV2-S pretrained on ImageNet, and the testing files, evaluating its performance based on class-wise accuracy and displaying misclassified images.

## Installation
To install the project, follow these steps:
```bash
# Clone repository
git clone https://github.com/asumak2003/Bottle_AnoDet
# Install Python and necessary libraries
pip install -r "requirements.txt"
```
Make sure to install the necessary libraries from the requirements file, such as PyTorch and Torchvision, to run the project.

## Structure
Below, the structure of the repository is represented as a tree diagram. It is worth noting that not all files are displayed, only the most essential ones that are actually required for the mantainance and further development of the project.

Bottle_AnoDet

├───EffNet_fine_tune.py

├───EffNet_test.py

├───scratch_model.py

├───live_classification.py

├───models

├───imgs

│   ├───bin_mask_opt.jpg

│   ├───data_loader

│   │   ├───fallen_after

│   │   ├───fallen_before

│   │   ├───no_anomaly

│   │   └───no_deckel

│   └───empty_rail

│       └───augumented

├───img_prep_utils

│   ├───img_agumentation.py

│   ├───img_prep.py

│   └───duplicate_removal

└───wrong_screwed_lid

    └───data

## Usage

#### Main Files
In the main folder, the files used for the training and testing and usage of the EfficientNet model can be found.
+ The [scratch_model.py](./scratch_model.py) file simply showcases an attempt to train a model from scratch on our data. 
+ The file [EffNet_fine_tune.py](./EffNet_fine_tune.py) is the main responsible for fine tuning the EfficientNet model, pretrained on ImageNet, on our dataset. It outputs a model and multiple graphs displaying the progress of the model during training. 
+ The output model can then be tested using [EffNet_test.py](./EffNet_test.py), displaying test loss, overall accuracy, class-wise accuracy and misclassified images for further analysis.
+ The file [live_classification.py](./live_classification.py) is used for direct inference on the live footage of the RTSP camera, by using one of the several models, found under the "models" folder. The script continuously processes frames and displays the prediction. The user can exit the live video feed by pressing 'q'.

#### Dataset and Image Preprocessing
Under the [imgs](./imgs/) folder, one may find the binary mask, augumented images, and the dataset (organised according to the requirements of the the PyTorch DataLoader function).

All files responsible for the preprocessing of images, mainly used to prepare images to be added to the dataset, are found under the [img_prep_utils](./img_prep_utils/) folder:
+ The [img_prep.py](./img_prep_utils/img_prep.py) file has multiple useful functions, including the sampling of videos, cropping and masking of images and duplicate removal. 
+ The [img_agumentation.py](./img_prep_utils/img_agumentation.py) file is used to create augumented images using brightness and contrast changes. 
+ Furthermore, the [duplicate_removal](./img_prep_utils/duplicate_removal/) folder has multiple files showcasing the different techniques explored for removing duplicated frames, where there has been no changes in the production chain from one image to the other.

#### Wrongly Screwed Lid
Finally, in the [wrong_screwed_lid](./wrong_screwed_lid/) folder, files attempting to detect when a lid has been misplaced can be found. Using a lot of data, it was concluded that this was not possible with our current equipment. The analysis of the collected data can be found under the [data](./wrong_screwed_lid/data/) folder.


## Contact
For any questions or feedback, please reach out to:
- **Email**: [imendezval@stud.hs-heilbronn.de](mailto:imendezval@stud.hs-heilbronn.de), [asumak@stud.hs-heilbronn.de](mailto:asumak@stud.hs-heilbronn.de)
- **GitHub Profile**: [imendezval](https://github.com/imendezval), [asumak2003](https://github.com/asumak2003)
- **LinkedIn**: [inigo-miguel-mendez-valero](https://www.linkedin.com/in/i%C3%B1igo-miguel-m%C3%A9ndez-valero-4ba3732b1/), [arian-sumak](https://www.linkedin.com/in/arian-sumak-6b5b8925a/)

Feel free to open an issue on GitHub or contact us in any way if you have any queries or suggestions.