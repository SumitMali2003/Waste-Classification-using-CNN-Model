# Waste-Management-Using-CNN-model

This project implements a waste classification system using Convolutional Neural Networks (CNNs) to classify waste images into categories like "Organic" and "Recyclable."

## Table of Contents

1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Output](#output)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Waste management plays an important role in reducing environmental pollution. This project uses a deep learning model to classify images of waste into two categories: **Organic** and **Recyclable**. We use Convolutional Neural Networks (CNNs), a popular model architecture for image classification tasks.

## Technologies Used

- **Python**: The primary programming language.
- **TensorFlow/Keras**: For building the deep learning model.
- **OpenCV**: For image preprocessing and manipulation.
- **Pandas**: For organizing and handling dataset.
- **Matplotlib**: For visualizing data and results.
- **NumPy**: For numerical operations.

## Dataset

The dataset used in this project contains images of waste items, categorized into two main classes:
- **Organic**
- **Recyclable**

The images are stored in the `TRAIN` and `TEST` directories under the folder `DATASET\DATASET`.

## Installation

1. Clone this repository to your local machine:
git clone https://github.com/yourusername/waste-management-cnn.git

2. Install the required dependencies:
pip install -r requirements.txt


Make sure to have the following libraries installed:

- TensorFlow
- OpenCV
- Pandas
- Matplotlib
- NumPy
- tqdm

3. Ensure that the dataset is placed in the correct directory structure

## Usage

To train the model, run the following Python script:
python train_model.py

This script will process the images in the `TRAIN` directory, visualize the distribution of the categories, and train a CNN model to classify waste items.

## Output

Upon running the script, you will see a pie chart visualizing the distribution of the images across the two categories, **Organic** and **Recyclable**. Below is an example output of the visualization:

![Uploading image.png…]()

### Example of Distribution of Categories


Organic: 52.36%
Recyclable: 47.64%

The pie chart helps you understand the balance of data between the two categories before training the model.

## Contributing

Feel free to fork this repository and submit a pull request. For any feature requests or bug reports, please open an issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
