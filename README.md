# Transfer Learning for Deep-Fake Video Classification: A VGG16-Based Study

**1.	MOTIVATION OF THE STUDY**

In recent years, the rise of deepfake technology has raised significant concerns regarding misinformation, identity fraud, and digital trust. Although visual content manipulation has become increasingly sophisticated, so have the methods to detect it. Our project addresses this challenge by developing a deepfake detection system based on Convolutional Neural Networks (CNNs) and transfer learning. Leveraging the power of VGG16, a robust pre-trained architecture, we aim to distinguish between real and fake face images extracted from video data. By utilizing transfer learning, we can benefit from the rich feature representations learned from large-scale datasets (like ImageNet) and apply them to a specific domain like deepfake detection — where datasets are often more limited in size and diversity. 
In our project, thousands of facial images that were extracted and cleaned from videos are used to prepare them for training. To prevent overfitting, data augmentation techniques were applied, and the training process was monitored using TensorBoard. The deep layers of the VGG16 model were fine-tuned to better adapt to our dataset. Model performance was evaluated using accuracy, F1-score, AUC-ROC, and a confusion matrix. Additionally, false positives and false negatives were analyzed to visualize and understand the model's weaknesses.

**2.	ARCH & HYPER-PARAMETERS**

In this study, a transfer learning approach was employed using the VGG16 architecture pre-trained on the ImageNet dataset as the base model. The goal was to fine-tune the model for binary classification. The architecture was modified by appending custom classification layers to the convolutional base of VGG16, enabling learning specific to the given dataset. After initially freezing the convolutional base of the VGG16 model to train only the custom classification head, all layers were later unfrozen for fine-tuning, allowing the entire network to be updated with a lower learning rate. 

A. CUSTOM HEAD

For the custom head, GlobalAveragePooling2D, which is used to reduces the feature map dimensions, and fully connected layer with 1024 units and ReLU activation were used. Also, dropout regularization is used to reduce overfitting with 0.5 drop rate. A fully connected output layer with a single neuron and a sigmoid activation function was used to perform binary classification, producing a probability score between 0 and 1 indicating the likelihood of the input belonging to the positive class.

B. IMAGE PREPROCESSING & AUGMENTATION

In the image preprocessing part, All images were resized to 224x224 and pixel values were normalized to the [0, 1] range. Data augmentation techniques, zoom and rescale, were applied to the training set to improve generalization and help the model learn invariant features. Layers are represented in the Figure 1.

<img width="176" alt="image" src="https://github.com/user-attachments/assets/a285f6d5-34ea-412a-a265-042882db1bc7" />

FIGURE 1. Architecture Details Of The Modified VGG16 Model

**3.	RESULTS AND DISCUSSION**

For the evaluation of the study, AUC-ROC, Accuracy and F1-score parameters are used. Results are shown in Table 1. 

TABLE 1. Evaluation Parameters

AUC-ROC	0.9743

Accuracy	0.9658

F1-score	0.8687

As a result of the evaluation part, we achieved an AUC-ROC of 0.9743, indicating that the model is highly capable of distinguishing between real and fake videos across different thresholds. The overall accuracy reached 96.58%, showing consistent and reliable classification. Our F1-score of 0.8687 reflects a balance between precision and recall, which is crucial for minimizing the misclassification of deep-fake content. Also, the Roc Curve is generated from results, as shown in the Figure 2.

<img width="204" alt="image" src="https://github.com/user-attachments/assets/a9356862-de01-40c0-86d9-a4969b77e392" />

FIGURE 2. ROC Curve Analysis

Also, a confusion matrix is created to analyze false positive and false positive rates, as shown in Figure 3.

<img width="228" alt="image" src="https://github.com/user-attachments/assets/5d232264-5dc6-4b06-b724-d4bdfaee3ccc" />

FIGURE 3. Confusion Matrix Evaluation 

The confusion matrix shows that the model correctly classified 3394 out of 3430 fake videos and 450 out of 550 real videos. It only made 36 false positive errors (real videos predicted as fake) and 100 false negative errors (fake videos predicted as real). This indicates that the model is especially good at detecting fake content, but slightly less accurate at recognizing real ones. Overall, the results show strong classification performance, especially in catching deep-fake content. 3 images for false negatives and 3 images for false positives are sampled as shown in the Figure 4 and Figure 5.

<img width="253" alt="image" src="https://github.com/user-attachments/assets/881e194e-4997-4093-b3b7-c9a30ec043b4" />

FIGURE 4. False Negative Samples

<img width="252" alt="image" src="https://github.com/user-attachments/assets/c39dcd9c-0a1f-46ae-9ddd-6166dc685341" />

FIGURE 5. False Positives Images

**4.	ETHICAL IMPACT OF DEEP-FAKE DETECTION**

Detecting deep-fake content is very important to protect public trust in digital media. Deep-fake videos can be used for misinformation, fraud, or harm to individuals. Therefore, a strong detection system helps reduce the spread of harmful content and supports digital safety. It increases people's confidence in the information they obtain through information technologies, thus supporting development. However, it is also important to consider privacy and the responsible use of detection tools. These systems should not be misused to monitor or control people unfairly.

**5.	LIMITATIONS OF THE MODEL**

In our study, the model showed high performance with an AUC-ROC of 0.9743 and an accuracy of 96.58%. However, as shown in the confusion matrix, the model misclassified 100 real videos as fake and 36 fake videos as real. This shows that it performs better on fake samples than on real ones. Limited training data or an imbalance between real and fake samples might also affect performance. Due to this reason, the model could produces worse results when detecting real samples.

# HOW TO RUN?



**1.Create and activate a virtual environment**

python -m venv venv

source venv/bin/activate

**2.Install project dependencies**

pip install -r requirements.txt

download image data from Kaggle: https://www.kaggle.com/datasets/greatgamedota/dfdc-part-34

**3.Before executing the notebook, make sure to update the file paths in the data pipeline section of the notebook**

Update these paths to match your local directory structure

image_root_dir = "./images"

csv_path = "./metadata34.csv"

Ensure these point to the correct location of your images and metadata CSV file.

**4.Running the Project**

the deepfake_new.ipynb notebook and run all cells sequentially.

This will:

-Load and preprocess the data,

-Train and evaluate the VGG16 model using transfer learning,

-Generate evaluation metrics including AUC-ROC, accuracy, F1-score,

-Visualize the ROC curve, confusion matrix, and sample false positives/negatives







