import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
DATA_PATH = "data/spam.csv"
if not os.path.exists(DATA_PATH):
    # If no local data, download from GitHub
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
else:
    df = pd.read_csv(DATA_PATH)

# Encode labels
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Load BERT sentence embedding model
print("🔍 Loading sentence transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
print("🔢 Creating embeddings...")
X_train_emb = embedder.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = embedder.encode(X_test.tolist(), show_progress_bar=True)

# Train logistic regression
print("⚙️ Training logistic regression classifier...")
model = LogisticRegression(max_iter=200)
model.fit(X_train_emb, y_train)

# Evaluate
y_pred = model.predict(X_test_emb)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Training complete! Accuracy: {acc:.2f}")

# Save
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(embedder, "model/bert_embedder.pkl")
print("💾 Model and embedder saved successfully.")
