# Spam SMS Detection

## Project Overview
This project aims to classify SMS messages as either spam or ham (not spam) using machine learning techniques. The dataset used for training contains labeled SMS messages, and the model learns to distinguish between legitimate and spam messages based on text features.

## Features
- Preprocessing of SMS messages (cleaning, tokenization, and vectorization)
- Implementation of multiple ML models (e.g., Naive Bayes, Logistic Regression, Random Forest)
- Performance evaluation using precision, recall, F1-score, and accuracy
- Deployment of the model via a web application or API

## Technologies Used
- Python
- Scikit-learn
- NLTK (Natural Language Toolkit)
- Pandas & NumPy
- Flask (for deployment, if applicable)
- Jupyter Notebook

## Dataset
The dataset consists of SMS messages labeled as spam or ham. It can be sourced from public datasets like the [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) or Kaggle.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-sms-detection.git
   cd spam-sms-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Model Training
To train the model, run the `train_model.py` script:
```bash
python train_model.py
```
This will preprocess the data, train the model, and save it for later use.

## Testing the Model
To test a new SMS message, use:
```bash
python predict.py "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121"
```

## Deployment (Optional)
To deploy the model using Flask, run:
```bash
python app.py
```
Then open `http://127.0.0.1:5000` in your browser to use the web interface.

## Performance Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

## Future Improvements
- Experimenting with deep learning models like LSTMs
- Implementing an ensemble learning approach
- Improving dataset quality with more labeled examples

## Contributing
Feel free to fork this repository and submit pull requests with improvements.



