Identification Fresh Thawed Beef CNN
In the meat industry, distinguishing between fresh and thawed beef is crucial but challenging. This project addresses this issue using Convolutional Neural Networks (CNNs) and InceptionV3 to accurately detect fresh and thawed beef. The goal is to prevent fraudulent practices that harm consumers and retailers by mixing meats of different qualities and nutritional values. 

The first model is a model where there is no preprocessing or data augmentation before the training process and the second model is a model that goes through preprocessing and data augmentation using the TensorFlow Keras ImageDataGenerator. 

The dataset used in CNN_Model_Beef comes from Mendeley data:
- LOCBEEF: Beef Quality Image dataset for Deep Learning Models (only take image data labeled fresh and cropped as Fresh data)
  - https://data.mendeley.com/datasets/nhs6mjg6yy/1

- Images of fresh and non-fresh beef meat samples (augmented from 64 images to 1600 images for Thawed data)
  - https://data.mendeley.com/datasets/wvhkpppddp/1

The dataset used in CNN_Model_keggle comes from Mendeley and keggle data:
- LOCBEEF: Beef Quality Image dataset for Deep Learning Models (only take image data labeled fresh and cropped as Fresh data)
  - https://data.mendeley.com/datasets/nhs6mjg6yy/1

- Meat Freshness Image Dataset (only take fresh and half fresh image label for thawed data)
  - https://www.kaggle.com/datasets/vinayakshanawad/meat-freshness-image-dataset

Key features of this project include:
- Data Splitting: 70% training, 15% validation, and 15% testing.
- Model Architecture: Utilizes Convolutional Neural Networks (CNNs) and InceptionV3 without hyperspectral imaging.
- Data Augmentation: Techniques such as image cropping and hard data generation improve model robustness.
- Optimization: Models were optimized using Adam and RMSprop, achieving high accuracy with low computational costs.
- Performance Metrics: Achieved up to 98.92% accuracy, precision, recall, and F1-score using InceptionV3 with Adam optimizer.
- Training Efficiency: The best model trained in approximately 18 minutes and 56 seconds.

Results:
- The first model with Adam optimization achieved 96.12% accuracy, while RMSprop achieved 98.06%.
- The second model with Adam achieved 97.84% accuracy, while RMSprop achieved 98.49%.
- The InceptionV3 model with Adam and RMSprop both achieved 98.92% accuracy.
- Compared to the highest accuracy results without a pre-trained model, the difference in accuracy was only 0.43%.

Technologies Used:
- Python: 3.11.5
- TensorFlow: 2.15.0

This repository provides the code and data necessary to replicate the results and improve the detection of fresh and thawed beef using advanced neural network techniques.
