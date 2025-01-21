# Applied Machine Learning Systems (ELEC0134) Assignment 24/25

## Project Overview
This project implements solutions for two machine learning tasks using datasets from MedMNIST:
1. **Task A (Binary Classification)**: Classify images from the BreastMNIST dataset into "Benign" or "Malignant".
2. **Task B (Multi-class Classification)**: Classify images from the BloodMNIST dataset into one of eight blood cell types.

## Project Structure
The repository is organized as follows:
- AMLS_24-25_SN21020841
  - A
    - all_models_a.py
    - best_model_A_final.pth
    - task_a_test.py
    - task_a_train.py
  - B
    - all_models_b.py
    - best_model_B_final.pth
    - task_b_test.py
    - task_b_train.py
  - data_prep.py
  - main.py
  - README.md

## Dataset
I created the `data_prep.py` file which imports and preprocesses data for Task A and Task B.  
It splits the data into batches using DataLoader to keep ready for model training and testing.  
I call the relevant data split in `task_x_train.py` and `task_x_test.py`. 

I used the medmnist API to download the BreastMNIST and BloodMNIST datasets in their predefined Training/Validation/Test splits.  
Dr Rodrigues clarified the `Dataset folder` is not needed in this case - as long as the model is kept blind to the test dataset - which I followed. 
  


## Role of Files
Folders A and B contain file with the same functionality, differing only in their naming convention. 'x' denotes the task name in the following explaination.

 - **all_models_x.py:**  
 It has all the models I tested before selecting the final one.
 It is called in the task_x_train.py file to select which model to train. The final models I selected are:  
  `BreastCancerClassifier9` (Task A)  
  `BloodCellClassifier1` (Task B) 

 - **best_model_X_final.pth:**  
 This is the final version of the selected model saved during the training process. It contains the learned weights and parameters from the best-performing epoch. This saved model represents the optimized and fully trained version, ready for testing.  
 It is called in the `task_x_test.py` file to run on the final test dataset.

- **task_x_test.py**:  
Script to test the final learned model on the test split of respective datsets.  
Function `run()` executes the saved best model `best_model_X_final.pth` on the test split.  
Outputs the confusion matrix (for Task A) and metric scores dataframe.

- **task_x_train.py**:  
Script to train and validate a model.  
Function `run()` executes the training loop with the predefined loss function and optimiser.  
You can choose the model from `all_models_x.py` to run.  
Outputs the confusion matrix (for Task A),training and validation loss/accuracy curves and best metric scores table. 

## How to Use
### Dependencies
You will need to have the following packages installed in your system:  
torch  
torchvision  
medmnist  
matplotlib  
pandas  
scikit-learn  
seaborn  

To install these you can run  
`pip install torch torchvision medmnist matplotlib pandas scikit-learn seaborn`
### main.py
You will only need to run the `main.py` file in the terminal.  
It will run some functions to prepare the datasets then display a menu. This is in a continuous loop and you will need to manually exit by entering the appropriate option.


1. **Task A**: Test Model  
   `best_model_A_final.pth` model is tested on the test split of BreastMNSIT dataset.

2. **Task A**: Visualize Model Training  
   Train the model chosen in `task_a_train.py` and visualise training and validation loss/accuracy.  
   Default choice is model BreastCancerClassifier9.  

3. **Task B**: Test Model  
   `best_model_B_final.pth` model is tested on the test split of BloodMNSIT dataset.

4. **Task B**: Visualize Model Training  
   Train the model chosen in `task_b_train.py` and visualise training and validation loss/accuracy.  
   Default choice is model BloodCellClassifier1.  

5. **Exit**:  
   Exit the loop.
