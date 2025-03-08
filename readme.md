# Political Speech Classification with Transformers

## Overview
An interactive application that identifies which politician delivered a speech segment using transformer-based deep learning. The model analyzes linguistic patterns to distinguish between speeches by Barack Obama, George W. Bush, and George H.W. Bush with 83% accuracy.

<img src="./NLP-Political Speech classification.png" alt="Demo" width="600">

## Features
- **Speech Classification**: Input any speech segment to identify the likely speaker
- **Attention Visualization**: Explore how transformers "pay attention" to relationships between words
- **Interactive Model Explorer**: Adjust model parameters and see their impact in real-time
- **Performance Analysis**: Visualize accuracy across different speech topics and contexts

## Technical Implementation
- **Custom Transformer Architecture**: Built from scratch in PyTorch with multi-head self-attention
- **End-to-End Pipeline**: From text tokenization to classification with NLTK and custom embeddings
- **Streamlit Interface**: Interactive visualizations for model explainability
- **Performance Optimization**: Efficient implementation suitable for browser-based inference

## Getting Started

### Installation
```bash
git clone https://github.com/jain-atishay/NLP-Political-Speech-Classification-with-Transformers.git
cd NLP-Political-Speech-Classification-with-Transformers
pip install -r requirements.txt
streamlit run app.py
```

### Usage
1. Navigate to the "Speaker Classification" section
2. Enter a speech segment in the text area
3. Click "Classify" to see the model's prediction
4. Explore other sections to understand the model's architecture and attention patterns

## Model Architecture
The system uses a transformer encoder with:
- 4 transformer layers with 2 attention heads each
- 64-dimensional token embeddings
- Absolute positional encoding
- Final classification through global average pooling

## Results
- **83% Test Accuracy** across three politicians
- Strongest performance on Obama's speeches (85.3% accuracy)
- Most common confusion: between G.W. Bush and G.H.W. Bush speeches
- Outperforms LSTM (76.5%) and traditional NLP approaches (63-72%)

## What I Learned
- Implementing transformer architectures from scratch
- Visualizing and interpreting self-attention mechanisms
- Building interactive ML demos with Streamlit
- Analyzing NLP model performance across different speech contexts

## Future Improvements
- Pre-training the encoder on a larger corpus before fine-tuning
- Experimenting with relative positional encodings
- Adding more politicians to the classification task
- Implementing cross-validation for more robust evaluation

## Live Demo
Try it here: [https://speech-classification-transformer.streamlit.app](https://speech-classification-transformer.streamlit.app)

## License
MIT License