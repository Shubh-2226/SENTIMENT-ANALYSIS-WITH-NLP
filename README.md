# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY NAME: CODETECH IT SOLUTIONS

NAME: NITIN MAHOR

ITERN ID: CT08RWJ

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH KUMAR

Here’s a detailed 650-word description of the Jupyter Notebook code:

---

### **Sentiment Analysis Using Naïve Bayes in Jupyter Notebook**
Sentiment analysis is a crucial task in natural language processing (NLP) that aims to determine the emotional tone behind a text. It is widely used in customer feedback analysis, social media monitoring, and recommendation systems. This Jupyter Notebook demonstrates a basic sentiment analysis pipeline using the Naïve Bayes classifier from Scikit-learn. The pipeline includes text preprocessing, feature extraction, model training, evaluation, and prediction. The implementation utilizes Python and various machine learning libraries to classify text as **positive, neutral, or negative** based on sentiment.

---

### **1. Data Preparation and Preprocessing**
The first step in the notebook is defining a sample dataset containing short text reviews with their corresponding sentiment labels. The dataset consists of a small set of manually labeled reviews, each classified as positive, neutral, or negative. This dataset is stored as a **Pandas DataFrame** for easy manipulation.

To enable numerical processing, the categorical sentiment labels are mapped to integer values:
- **Positive sentiment → 1**
- **Neutral sentiment → 0**
- **Negative sentiment → -1**

Before training the model, the dataset is split into training and test sets using `train_test_split()` from Scikit-learn. This ensures that the model is evaluated on unseen data, preventing overfitting.

---

### **2. Feature Extraction**
Since machine learning models cannot directly process raw text, the text data must be converted into numerical features. This is accomplished through a two-step transformation using **CountVectorizer** and **TfidfTransformer**:

1. **CountVectorizer**: This transforms text into a word frequency matrix, where each row represents a document and each column represents a unique word’s occurrence count.
2. **TfidfTransformer**: This applies Term Frequency-Inverse Document Frequency (TF-IDF) weighting, which helps highlight important words while reducing the influence of commonly used terms.

These feature extraction steps are combined into a **Scikit-learn Pipeline**, allowing streamlined data preprocessing before feeding into the classifier.

---

### **3. Model Training**
The model used for sentiment classification is the **Multinomial Naïve Bayes (MultinomialNB)** algorithm. It is particularly effective for text classification tasks due to its probabilistic nature and ability to handle sparse data efficiently. The Naïve Bayes classifier makes predictions by calculating the probability of a text belonging to a specific class, given the word distribution.

The pipeline is trained using the `fit()` function on the training dataset (`X_train` and `y_train`). After training, the model is ready to predict sentiment from new text data.

---

### **4. Model Evaluation**
After training, the model's performance is evaluated using several metrics:

- **Accuracy Score**: Measures the percentage of correct predictions.
- **Classification Report**: Provides precision, recall, and F1-score for each sentiment class.
- **Confusion Matrix**: Displays a matrix comparing predicted vs. actual labels.

The **confusion matrix** is visualized using the Seaborn library, making it easy to interpret the model’s performance. The `classification_report()` function includes the `zero_division=0` parameter to handle cases where there are no predicted samples for a particular class.

---

### **5. Real-Time Sentiment Prediction**
A function, `predict_sentiment()`, is defined to classify new text input. This function takes a string as input, processes it through the trained model, and returns the predicted sentiment label (Positive, Neutral, or Negative). 

For demonstration, the model is tested with an unseen sentence:  
*"This product exceeded my expectations!"*  
The function predicts the sentiment and prints the result.

---

### **6. Tools and Technologies Used**
This notebook utilizes several Python libraries for data manipulation, machine learning, and visualization:

- **Pandas**: For handling tabular data.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For plotting the confusion matrix.
- **Scikit-learn**: For machine learning, including vectorization, Naïve Bayes classification, and model evaluation.

---

### **7. Jupyter Notebook Environment**
This code is designed to run in a **Jupyter Notebook**, an interactive computing platform widely used in data science and machine learning. It can be executed in **JupyterLab**, **Google Colab**, or any local Python environment that supports notebooks.

---

### **Conclusion**
This project provides a simple yet effective implementation of sentiment analysis using Naïve Bayes in a Jupyter Notebook. By leveraging Scikit-learn’s powerful tools, the notebook efficiently processes text, builds a sentiment classification model, and evaluates its performance. This approach is highly scalable and can be extended with larger datasets or deep learning models like LSTMs or transformers for improved accuracy.

