# Emotion-Classification-project


🎧 MARS Project — Speech Emotion Classification
This deep learning-based project aims to classify human emotions from raw speech audio files. The system leverages a hybrid model combining CNN (Convolutional Neural Networks), GRU (Gated Recurrent Units), and a custom Attention Layer for context-aware emotion recognition.

📁 Dataset: RAVDESS
Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Total Audio Files: ~2500 .wav files

Emotions Covered:
neutral, calm, happy, sad, angry, fearful, disgust, surprised

🎛️ Data Preprocessing
Feature Extraction:
Extracted 40 MFCC (Mel-Frequency Cepstral Coefficients) per time step using librosa.

Padding:
All MFCC matrices were padded or truncated to a fixed length of 200 time steps.

Data Augmentation:
To mitigate data scarcity and improve generalization:

Applied 4× augmentation using:

Time stretching

Pitch shifting

Noise addition

🧠 Model Architecture
Layer	Details
Conv1D	Extracts local temporal features
Conv1D	Further captures speech patterns
GRU	Learns sequential dependencies
Attention	Highlights important time steps
Dense (Output)	Outputs emotion probabilities

Activation: Softmax

Loss Function: Categorical Crossentropy

Optimizer: Adam
![WhatsApp Image 2025-06-25 at 20 21 51_4aa6bb8c](https://github.com/user-attachments/assets/d1692343-50f2-4b84-aa18-447bd620d744)





📈 Model Training
Epochs: 100
Best Accuracy Achieved: 81% 
Overall Accuracy: 81%

Macro Avg F1 Score: 81%

Weighted Avg F1 Score: 81%

📝 Note: Classes like **fearful** and **Sad** showed slightly lower performance, likely due to limited data. Can be improved with more samples.

![WhatsApp Image 2025-06-25 at 20 23 06_4131849d](https://github.com/user-attachments/assets/aaa439bf-a221-4f33-90c4-ac790fa90dd8)

![WhatsApp Image 2025-06-25 at 20 23 44_8e2d9b45](https://github.com/user-attachments/assets/00700eb5-3a97-4449-8ece-31f96ab7721e)

🌐 Web Application
Built using Streamlit

Deploys the trained model for real-time emotion classification

Allows users to upload .wav files (2–3 seconds)

Outputs predicted emotion

🎥 Preview video attached 

👉 Live App (Streamlit Cloud)  https://emotion-classifier-app-webapp-pesfcqbszywu2p454gva78.streamlit.app/

🛠️ Tech Stack
Libraries:

TensorFlow, Keras, Librosa, NumPy, Pandas, Matplotlib, Streamlit

Tools & Platforms:

Google Colab — Model development and training

VS Code — Code editing and version control

Streamlit Cloud — Web app deployment

✅ Conclusion
This project demonstrates the power of combining audio signal processing with deep learning architectures to build emotion-aware AI systems. It can be extended to real-time applications such as:

Virtual assistants

Mental health monitoring

Customer support sentiment analysis
