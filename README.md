# Fake News Detection using Deep Learning

## 📋 Project Overview

This project implements a **Fake News Detection System** using deep learning techniques to classify news articles as either **Real** or **Fake**. The system combines traditional machine learning approaches (TF-IDF vectorization) with deep neural networks to achieve high accuracy in detecting misinformation.

**IBM Project Submission** - Internal Assessment (60 Marks)  
**Course:** Deep Learning  
**Environment:** Google Colab  
**Dataset:** Kaggle Fake News Dataset

## 🎯 Objective

The primary goal is to develop an automated system that can:
- Analyze news article content (title + text)
- Extract meaningful features using TF-IDF vectorization
- Classify articles as Real (1) or Fake (0) using deep neural networks
- Provide reliable predictions for new, unseen news articles

## 📊 Dataset Information

#### Source: [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

#### Files Used:
- `Fake.csv` - Contains fake news articles
- `True.csv` - Contains real news articles

**Dataset Structure:**
- **Title:** News article headline
- **Text:** Full article content
- **Label:** 0 (Fake) or 1 (Real)

## 🛠️ Implementation

1. **Data Preprocessing**
   - Text cleaning and normalization
   - Stopword removal
   - URL and special character removal
   - Content combination (title + text)

2. **Feature Engineering**
   - TF-IDF Vectorization (max_features=5000)
   - Text-to-numerical conversion

3. **Deep Learning Model**
   - **Input Layer:** TF-IDF features (5000 dimensions)
   - **Hidden Layer 1:** 512 neurons + ReLU activation + Dropout (0.3)
   - **Hidden Layer 2:** 256 neurons + ReLU activation + Dropout (0.3)
   - **Output Layer:** 1 neuron + Sigmoid activation (binary classification)

4. **Training Configuration**
   - **Optimizer:** Adam
   - **Loss Function:** Binary Crossentropy
   - **Metrics:** Accuracy
   - **Epochs:** 5
   - **Batch Size:** 64
   - **Validation Split:** 20%

## 💻 Running the Project

#### Option 1: Google Colab (Recommended)
1. Upload the Jupyter notebook (`FakeNewsProject.ipynb`) to Google Colab
2. Upload the dataset files when prompted
3. Run all cells sequentially

#### Option 2: Local Environment
1. Open `FakeNewsProject.ipynb` in Jupyter Notebook/Lab
2. Ensure dataset files are in the correct path
3. Execute all cells

## 📈 Model Performance

The model achieves high accuracy in distinguishing between real and fake news articles through:

- **Text Preprocessing:** Comprehensive cleaning and normalization
- **Feature Extraction:** TF-IDF vectorization capturing important word patterns
- **Deep Learning:** Multi-layer neural network with dropout for regularization
- **Evaluation Metrics:** Classification report and confusion matrix analysis

## 🔍 Key Features

### 1. **Comprehensive Text Preprocessing**
- Lowercase conversion
- URL removal
- Punctuation and number removal
- Stopword filtering
- Extra whitespace normalization

### 2. **Advanced Feature Engineering**
- TF-IDF vectorization with optimal feature selection
- Content combination (title + text) for better context

### 3. **Deep Neural Network**
- Multi-layer architecture with dropout regularization
- Binary classification with sigmoid activation
- Adam optimizer for efficient training

### 4. **Visualization and Analysis**
- Training/validation accuracy plots
- Word clouds for fake vs real news
- Distribution analysis of dataset
- Performance metrics visualization

## 📁 Project Structure

```
FakeNewsProject/
│
├── FakeNewsProject.ipynb    # Main Jupyter notebook
├── README.md                # Project documentation
├── archive (3).zip          # Dataset archive
├── Fake.csv                 # Fake news dataset
├── True.csv                 # True news dataset
```

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Deep Learning:** Neural network architecture design and training
- **Natural Language Processing:** Text preprocessing and feature extraction
- **Machine Learning:** Classification, model evaluation, and validation
- **Data Science:** Data analysis, visualization, and interpretation
- **Python Programming:** Libraries like TensorFlow, scikit-learn, pandas

## 📊 Results and Insights

The project successfully demonstrates:
- Effective fake news detection using deep learning
- Importance of text preprocessing in NLP tasks
- Power of combining traditional ML (TF-IDF) with deep learning
- Practical application of neural networks in real-world problems



## 🙏 Acknowledgments

- **IBM** for providing the deep learning course and project opportunity
- **Kaggle** for the fake news dataset
- **TensorFlow/Keras** community for excellent documentation
- **Open Source Community** for the various libraries used

---

**Note:** This project is submitted as part of IBM's Deep Learning course internal assessment. The implementation showcases practical application of deep learning concepts in solving real-world problems like misinformation detection.
