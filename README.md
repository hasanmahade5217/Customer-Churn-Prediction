# Customer-Churn-Prediction

####Introduction:
This project aims to predict customer churn using Artificial Neural Networks (ANN). Predicting customer churn is crucial for businesses to retain customers and improve customer satisfaction.

####Dataset Description:
The dataset used for this project contains information about customer demographics, usage patterns, and churn status. It includes features such as customer age, subscription type, usage frequency, and churn status (whether the customer churned or not).

####Data Preprocessing:
- Cleaned the dataset by handling missing values and removing duplicates.
- Encoded categorical variables using one-hot encoding.
- Scaled numerical features using Min-Max scaling.
####Model Architecture:
- Built a feedforward neural network with an input layer, multiple hidden layers, and an output layer.
- Used ReLU activation function for hidden layers and sigmoid activation function for the output layer.
- Implemented binary cross-entropy loss function and Adam optimizer.
####Training the Model:
- Split the dataset into training and validation sets (80/20 split).
- Trained the ANN model using batch size of 32, learning rate of 0.001, and 50 epochs.
#### Model Evaluation:
- Evaluated the model's performance using accuracy, precision, recall, and F1-score.
- Visualized the confusion matrix to analyze the model's predictions.
#### Results:
- Achieved an accuracy of 85% on the validation set.
- Identified key features contributing to customer churn, such as subscription type and usage frequency.

####Deployment:
- Explored options for deploying the trained model in production environments.
- Considered integrating the model into existing customer relationship management systems.

####Usage:
To replicate the churn prediction process:

- Install the required dependencies listed in requirements.txt.
- Run the train_model.py script to train the ANN model.
- Use the trained model to make predictions on new customer data.

####Future Improvements:
- Experiment with different neural network architectures (e.g., adding dropout layers).
- Incorporate additional features such as customer feedback or satisfaction scores.
- Explore ensemble learning techniques to improve model performance further.

####Contributing:
Contributions are welcome! Please feel free to open an issue or submit a pull request with any improvements or suggestions.


#### Acknowledgments:
We thank [Dataset source] for providing the dataset used in this project.
