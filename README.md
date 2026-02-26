# Sentiment Analysis with Neural Networks

**Deep learning approach to sentiment classification.**

## What It Does

Implements neural network models for sentiment analysis on text data. Trains on labeled datasets and predicts positive/negative/neutral sentiment.

**Why it matters:** Traditional ML struggles with sarcasm and context. Neural networks capture nuanced patterns.

## Quick Start

\`\`\`bash
# Clone
git clone https://github.com/avishek15/Sentiment-with-Neural-network.git
cd Sentiment-with-Neural-network

# Install
pip install -r requirements.txt

# Train
python train.py --dataset imdb --epochs 10

# Predict
python predict.py --text "This movie was amazing!"
\`\`\`

## Features

- **Multiple architectures** — LSTM, GRU, Transformer
- **Pre-trained models** — Use without training
- **Custom datasets** — Train on your data
- **Evaluation metrics** — Accuracy, F1, confusion matrix

## Tech Stack

- **Python 3.8+**
- **PyTorch** — Deep learning framework
- **Transformers** — Hugging Face models
- **NLTK** — Text preprocessing

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| LSTM | 88.5% | 0.88 |
| GRU | 89.2% | 0.89 |
| BERT (fine-tuned) | 94.1% | 0.94 |

## License

MIT License - see [LICENSE](LICENSE)

## Author

Built by [Avishek Majumder](https://invaritech.ai)
