!pip install transformers datasets scikit-learn matplotlib seaborn -q

#Import libraries
from transformers import pipeline
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

print("✅ All libraries imported successfully!")

# Load the model
print("Loading model...")
classifier = pipeline("sentiment-analysis",
                     model="distilbert-base-uncased-finetuned-sst-2-english")
print("✅ Model loaded successfully!")

# Load IMDB dataset (we'll use a small subset for speed)
print("Loading dataset...")
dataset = load_dataset("imdb", split="test")  # First 1000 test examples
dataset  = dataset.shuffle (seed=42).select(range(1000))
print(f"✅ Dataset loaded {len(dataset)} reviews ready for evaluation!")

# Verify balance
label_counts = Counter(dataset['label'])
print(f"...Negative reviews: {label_counts[0]}")
print(f"...Positive reviews: {label_counts[1]}")

# Function to predict on batch
def predict_sentiment(texts, batch_size=32):
    """
    Predict sentiment for a list of texts in batches.
    """

    predictions = []
    totalbatches = (len(texts) + batch_size - 1) // batch_size

    print(f"Processing {len(texts)} reviews in {totalbatches}) batches...")

    for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
        batch = texts[i:i+batch_size]
        results = classifier(batch, truncation=True, max_length=512)
        predictions.extend(results)

        #Progress indicator
        if batch_num % 10 == 0 or batch_num == totalbatches:
            print(f"Processed batch {batch_num}/{totalbatches} ")

    return predictions

# Get predictions
print("Making predictions on 1000 reviews...   ")
texts = dataset['text']
predictions = predict_sentiment(texts)

#Convert predictions to binary lables
pred_labels = [1 if p['label'] == 'POSITIVE' else 0 for p in predictions]
true_labels = list(dataset['label'])

print(f"\n✅ Converted {len(pred_labels)} predictions to binary labels.")

# Determine accuracy and evaluation results
accuracy = accuracy_score(true_labels, pred_labels)

print("="*70)
print("EVALUATION RESULTS")
print("="*70)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Correct Predictions: {sum(np.array(true_labels) == np.array(pred_labels))} / {len(true_labels)}")

print("\nDetailed Classification Report:")
print(classification_report(true_labels, pred_labels,
                            target_names=['Negative', 'Positive']))
							
# Calculate confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Create visualisation
plt.figure(figsize=(8, 6))

sns.heatmap(cm,
           annot=True,
           fmt='d',
           cmap='Blues',
           xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'],
           cbar_kws={'label': 'Number of Reviews'},
           annot_kws={'size': 16,'weight': 'bold'})

plt.xlabel('Predicted Sentiment', fontsize=12, weight='bold')
plt.ylabel('Actual Sentiment', fontsize=12, weight='bold')
plt.title('Confusion Matrix - Sentiment Analysis Results\n' +
          f'Accuracy: {accuracy:.2%}',
          fontsize=14, weight='bold',pad=20)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300,bbox_inches='tight')
plt.show()

print("✅ Confusion matrix saved as 'confusion_matrix.png'!")

# Extract values from confusion matrix
tn, fp, fn, tp = cm.ravel()

print("="*70)
print("CONFUSION MATRIX BREAKDOWN")
print("="*70)
print(f"\nTrue Negatives (TN): {tn}")
print(f" ✅ Correctly identified as NEGATIVE: {tn} reviews")

print(f"\nFalse Positives (FP): {fp}")
print(f" ❌ Incorrectly identified as POSITIVE: {fp} reviews")

print(f"\nFalse Negatives (FN): {fn}")
print(f" ❌ Incorrectly identified as NEGATIVE: {fn} reviews")

print(f"\nTrue Positives (TP): {tp}")
print(f" ✅ Correctly identified as POSITIVE: {tp} reviews")

print("\n")
print("KEY INSIGHTS:")
print("=" * 70)

# Calculate error rates
false_positive_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
false_negative_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0

print(f"\n False Positive Rate: {false_positive_rate:.2f}%")
print(f"  (Model incorreclty calls negative reviews positive)")

print(f"\n False Negative Rate: {false_negative_rate:.2f}%")
print(f"  (Model incorrectly calls positive reviews negative)")

if false_positive_rate > false_negative_rate:
    print("\n⚠️ The model tends to be OPTIMISTIC (over predicted postive).")
elif false_negative_rate > false_positive_rate:
    print("\n⚠️ The model tends to be PESSIMISTIC (over predicted negative).")  
else:
    print("\n✅ The model has a BALANCED in its errors.")

print("="*70)
print("EXAMPLE MISCLASSIFICATIONS")
print("="*70)

#Find misclassified examples
errors = []
for i, (true, pred) in enumerate(zip(true_labels, pred_labels)):
    if true != pred:
        errors.append({
            'index': i,
            'text':texts[i],
            'true':'Positive' if true == 1 else 'Negative',
            'predicted':'Positive' if pred == 1 else 'Negative',
            'confidence': predictions[i]['score']            
        })

print(f"\nFound {len(errors)} misclassified reviews out of {len(true_labels)}")
print(f"Error rate: {len(errors)/len(true_labels)*100:.2f}%")

#Show high-confidence errors (most interesting)
high_conf_errors = [e for e in errors if e['confidence'] > 0.9]
print(f"\nHigh-confidence errors (>90% confidence): {len(high_conf_errors)}")

print("\n" + "-"*70)
print("TOP 5 MISCLASSIFIED (Most confidence mistakes)")
print("-"*70)

for i,error in enumerate(sorted(high_conf_errors,
                              key=lambda x: x['confidence'], 
                              reverse=True)[:5],1):
    print(f"\n[i]. Review (first 200 chars):")
    print(f"    {error['text'][:200]}...")
    print(f"    ✓ Actual: {error['true']}")
    print(f"    ✘ Predicted: {error['predicted']} (confidence: {error['confidence']:.2%})")
