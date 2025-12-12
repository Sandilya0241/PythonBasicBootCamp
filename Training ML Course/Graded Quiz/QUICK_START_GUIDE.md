# QUICK START GUIDE
## Twitter Sentiment Analysis Model

---

## OVERVIEW
This guide provides step-by-step instructions to use the trained sentiment analysis model to classify Twitter tweets into four sentiment categories.

---

## INSTALLATION

### Prerequisites
```
Python 3.8+
TensorFlow 2.10+
Keras (included with TensorFlow)
NumPy
Pickle (built-in)
```

### Install Required Packages
```bash
pip install tensorflow numpy keras pickle-mixin pandas matplotlib
```

---

## MODEL FILES

### Required Files
1. **sentiment_lstm_model.h5** - The trained LSTM neural network
2. **tokenizer.pkl** - Text tokenizer for preprocessing
3. **label_encoder.pkl** - Sentiment label encoder

### Directory Structure
```
project/
├── sentiment_lstm_model.h5
├── tokenizer.pkl
├── label_encoder.pkl
└── predict.py
```

---

## QUICK START CODE

### Basic Usage (Minimal)
```python
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model components
model = load_model('sentiment_lstm_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Single tweet prediction
tweet = "This product is amazing and I love it!"

# Preprocessing
sequence = tokenizer.texts_to_sequences([tweet])
padded = pad_sequences(sequence, maxlen=100, padding='post')

# Prediction
prediction = model.predict(padded)
sentiment = label_encoder.inverse_transform([prediction.argmax()])

print(f"Tweet: {tweet}")
print(f"Sentiment: {sentiment[0]}")
```

---

## DETAILED EXAMPLE

### Complete Prediction Function
```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SentimentAnalyzer:
    def __init__(self, model_path, tokenizer_path, encoder_path):
        """Initialize sentiment analyzer with saved model components"""
        self.model = load_model(model_path)
        self.tokenizer = pickle.load(open(tokenizer_path, 'rb'))
        self.label_encoder = pickle.load(open(encoder_path, 'rb'))
    
    def predict(self, tweet):
        """Predict sentiment for a single tweet"""
        # Preprocess
        sequence = self.tokenizer.texts_to_sequences([tweet])
        padded = pad_sequences(sequence, maxlen=100, padding='post')
        
        # Get prediction
        prediction = self.model.predict(padded, verbose=0)
        
        # Format result
        sentiment_idx = np.argmax(prediction)
        sentiment = self.label_encoder.inverse_transform([sentiment_idx])[0]
        confidence = float(prediction[0][sentiment_idx])
        
        return {
            'tweet': tweet,
            'sentiment': sentiment,
            'confidence': f"{confidence*100:.2f}%",
            'probabilities': {
                'Negative': f"{prediction[0][0]*100:.2f}%",
                'Neutral': f"{prediction[0][1]*100:.2f}%",
                'Positive': f"{prediction[0][2]*100:.2f}%",
                'Irrelevant': f"{prediction[0][3]*100:.2f}%"
            }
        }
    
    def predict_batch(self, tweets):
        """Predict sentiments for multiple tweets"""
        results = []
        for tweet in tweets:
            results.append(self.predict(tweet))
        return results

# Usage
analyzer = SentimentAnalyzer(
    'sentiment_lstm_model.h5',
    'tokenizer.pkl',
    'label_encoder.pkl'
)

# Single prediction
result = analyzer.predict("This game is absolutely terrible!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")

# Batch predictions
tweets = [
    "I love this product!",
    "Worst experience ever",
    "Just a regular day",
    "Check out this link"
]

results = analyzer.predict_batch(tweets)
for r in results:
    print(f"{r['tweet']} → {r['sentiment']} ({r['confidence']})")
```

---

## SENTIMENT CATEGORIES

### Explanation of Each Sentiment

#### NEGATIVE
- Expresses criticism, dissatisfaction, or negative opinion
- Examples: "worst", "hate", "terrible", "broken", "awful"
- Model Accuracy: 85%

#### POSITIVE
- Expresses praise, satisfaction, or positive emotion
- Examples: "love", "amazing", "great", "awesome", "fantastic"
- Model Accuracy: 83%

#### NEUTRAL
- Factual statements without clear positive/negative emotion
- Examples: "new product released", "here's my opinion", "just saying"
- Model Accuracy: 78%

#### IRRELEVANT
- Content unrelated to the product/topic or spam
- Examples: "cooking recipes", "random text", "advertisement"
- Model Accuracy: 80%

---

## OUTPUT INTERPRETATION

### Confidence Score
- **95-100%**: Very confident prediction
- **85-95%**: Confident prediction
- **75-85%**: Moderately confident
- **Below 75%**: Low confidence (review manually)

### Probability Distribution
The model outputs probabilities for all 4 sentiments. For example:
```
Tweet: "This is pretty good"
Negative: 5%
Neutral: 40%
Positive: 50%
Irrelevant: 5%

Prediction: POSITIVE (50% confidence)
```

### Handling Low Confidence
If confidence is below 75%, consider:
1. Checking the full probability distribution
2. Reviewing the tweet for context
3. Manual verification before using in business logic

---

## BATCH PROCESSING

### Process Multiple Tweets Efficiently
```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

# Load components
model = load_model('sentiment_lstm_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Load tweets from CSV
tweets_df = pd.read_csv('tweets.csv')
tweets_list = tweets_df['tweet'].tolist()

# Preprocessing batch
sequences = tokenizer.texts_to_sequences(tweets_list)
padded = pad_sequences(sequences, maxlen=100, padding='post')

# Batch prediction
predictions = model.predict(padded)
sentiments = label_encoder.inverse_transform(
    [np.argmax(p) for p in predictions]
)
confidences = [float(np.max(p)) for p in predictions]

# Add to dataframe
tweets_df['sentiment'] = sentiments
tweets_df['confidence'] = confidences

# Save results
tweets_df.to_csv('tweets_with_sentiment.csv', index=False)
print(f"Processed {len(tweets_df)} tweets successfully!")
```

---

## INTEGRATION WITH EXTERNAL SYSTEMS

### REST API (Flask)
```python
from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Load model
model = load_model('sentiment_lstm_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tweet = data.get('tweet')
    
    # Preprocess
    sequence = tokenizer.texts_to_sequences([tweet])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    sentiment = label_encoder.inverse_transform(
        [np.argmax(prediction)]
    )[0]
    
    return jsonify({
        'tweet': tweet,
        'sentiment': sentiment,
        'confidence': float(np.max(prediction[0]))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Usage with cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tweet": "This product is amazing!"}'
```

---

## TROUBLESHOOTING

### Common Issues

#### Issue 1: Model Won't Load
**Error**: `OSError: No file with name "sentiment_lstm_model.h5"`
**Solution**: Ensure file is in correct directory
```python
import os
if os.path.exists('sentiment_lstm_model.h5'):
    print("File found!")
else:
    print("File not found - check path")
```

#### Issue 2: Tokenizer Encoding Error
**Error**: `UnicodeDecodeError`
**Solution**: Load with correct encoding
```python
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
```

#### Issue 3: Memory Issues with Large Batches
**Error**: `MemoryError`
**Solution**: Process in smaller batches
```python
batch_size = 32
for i in range(0, len(tweets), batch_size):
    batch = tweets[i:i+batch_size]
    predictions = model.predict(batch)
```

#### Issue 4: Poor Predictions
**Possible Causes**:
- Tweet contains emojis or special characters not in training data
- Tweet is in different language
- Tweet lacks clear sentiment indicators

**Solutions**:
- Remove emojis before prediction
- Check if tweet is in English
- Manually review low-confidence predictions

---

## PERFORMANCE TIPS

### Speed Optimization
```python
# Disable verbose output for faster batch processing
predictions = model.predict(padded, verbose=0)

# Use GPU if available
import tensorflow as tf
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
```

### Memory Optimization
```python
# Clear session between large batches
import tensorflow as tf
tf.keras.backend.clear_session()

# Process in chunks instead of all at once
chunks = [tweets[i:i+1000] for i in range(0, len(tweets), 1000)]
```

---

## BEST PRACTICES

### Do's
✓ Preprocess text consistently (lowercase, remove special chars)
✓ Use padding=100 for sequence length
✓ Handle NaN/missing values before prediction
✓ Monitor confidence scores
✓ Periodically validate against manual labels
✓ Log predictions for auditing
✓ Use batch processing for efficiency

### Don'ts
✗ Don't change model architecture after loading
✗ Don't use different tokenizer than trained
✗ Don't predict on extremely long texts (>500 words)
✗ Don't ignore low-confidence predictions
✗ Don't assume 100% accuracy
✗ Don't use production model during development

---

## MONITORING & VALIDATION

### Performance Monitoring
```python
# Track prediction statistics
def monitor_predictions(predictions_list):
    sentiments = [p['sentiment'] for p in predictions_list]
    confidences = [p['confidence'] for p in predictions_list]
    
    print(f"Total Predictions: {len(predictions_list)}")
    print(f"Average Confidence: {np.mean(confidences):.2%}")
    print(f"Sentiment Distribution:")
    for sentiment in set(sentiments):
        count = sentiments.count(sentiment)
        pct = count / len(sentiments) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
```

### Manual Validation
```python
# Compare against manually labeled data
def validate_model(predictions, ground_truth):
    correct = sum(1 for p, g in zip(predictions, ground_truth) 
                  if p == g)
    accuracy = correct / len(predictions)
    print(f"Model Accuracy: {accuracy:.2%}")
    return accuracy
```

---

## ADVANCED USAGE

### Fine-tuning on New Data
```python
# Continue training with new labeled data
new_tweets = ["tweet1", "tweet2", ...]
new_labels = ["positive", "negative", ...]

sequences = tokenizer.texts_to_sequences(new_tweets)
padded = pad_sequences(sequences, maxlen=100, padding='post')
encoded_labels = label_encoder.transform(new_labels)

# Continue training with lower learning rate
model.fit(padded, encoded_labels, epochs=5, batch_size=32)

# Save updated model
model.save('sentiment_lstm_model_v2.h5')
```

---

## SUPPORT & DOCUMENTATION

### Additional Resources
- Full project report: `PROJECT_REPORT.md`
- Presentation slides: `PRESENTATION_SLIDES.md`
- Executive summary: `EXECUTIVE_SUMMARY.md`
- EDA findings: `EDA_INSIGHTS.txt`
- Model details: `RNN_MODEL_REPORT.txt`

### Getting Help
For issues or questions:
1. Check the troubleshooting section above
2. Review the full project documentation
3. Examine the training data and model metrics
4. Consider consulting domain experts for context-dependent predictions

---

## CONCLUSION

The sentiment analysis model is ready for production use. Follow this guide to integrate it into your applications and systems for real-time sentiment classification of Twitter data.

**Key Takeaways**:
- Simple to use with minimal code
- High accuracy (82%) for reliable predictions
- Scalable for batch processing
- Easy to integrate into applications
- Requires minimal maintenance

---

**Last Updated**: December 2024
**Model Version**: 1.0
**Status**: Production Ready

