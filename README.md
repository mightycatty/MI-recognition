# MI Recognition

This project involves using Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to classify motor imagery in EEG signals (Brain-Computer Interface, BCI).
- The CNN model was unable to converge during training.
- The RNN model achieved a test accuracy of approximately 60%.

## Usage

```bash
# 1. Preprocess the data
python utils/preprocess.py

# 2. Train the RNN model
python models/train_rnn.py

# 3. Evaluate the RNN model
python models/evaluate_rnn.py
