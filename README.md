# Unsupervised detection of anomalous sounds

in the context of AI-based factory automation, the objective of this project is to automatically identify whether the sounds produced by industrial machines are normal or anomalous (faulty machines). This is crucial for ensuring efficient and safe operations 
Goal is to detect unknown anomalous sounds under the condition that only normal sound samples have been provided as training data.


## Data and Preprocessing
The dataset used is a subset of [Malfunctioning Industrial Machine Investigation And Inspection](https://arxiv.org/abs/1909.09347) aka MIMII dataset (Detection and Classification of Acoustic Scenes and Events 2019 Workshop)

<p align="center">
  <img src="https://github.com/Berhanetek/unsupervised-detection-of-anomalous-sounds/assets/60297609/34c826a4-a2e1-43e8-a107-228337c16f20" width="600">
</p>

### Spectrograms
Preprocessing the gived audio files is a crucial step before diving deep into any modeling. Using the raw audio files comes with some problems some of which are:
 - Temporal Invariance: Models trained directly on raw waveforms may struggle with variations in timing and duration. The same sound might occur at different points in time, and the model should recognize it regardless of when it happens.
 - High Dimensionality: Raw audio waveforms are continuous signals with high temporal resolution. They consist of amplitude values sampled at very fine time intervals. This high-dimensional data can be computationally intensive to process directly

The approach we took is to convert the raw audio files and convert them into spectrograms so we can go ahead with the problem as if it's a vision problem.

Why Spectrograms?
- Feature Extraction: Spectrograms provide a more compact and structured representation of audio signals. Instead of working with raw audio waveforms, which can be complex and difficult to analyze directly, spectrograms capture the frequency content of audio over time. This representation is often easier for models to process.
- Feature Extraction: Spectrograms inherently capture important features such as pitch, rhythm, and timbre, which are crucial for audio understanding. Deep learning models can learn these features more effectively from spectrograms
- Noise Filtering: Spectrograms can visually highlight patterns corresponding to the actual audio content while dampening the impact of constant or repetitive noise, as it appears as consistent energy across frequencies. This can make it easier for machine learning models to focus on the relevant audio features

To accomplish this we used librosa, an audio processing package

<p align="center">
  <img src="https://github.com/Berhanetek/unsupervised-detection-of-anomalous-sounds/assets/60297609/f7fd5107-b485-4035-8e62-e1dbb479805b" width="600">
</p>

### Mixup
Another technique used is [Mixup](https://arxiv.org/abs/1710.09412). The idea behind it is interpolation: by combining two samples, mixup creates a new data point that lies between the original examples in feature space. This is good for smoother decision boundaries, improved generalization, learning of invariant and discriminative features.


<p align="center">
  <img src="https://github.com/Berhanetek/unsupervised-detection-of-anomalous-sounds/assets/60297609/e999905c-f050-443b-b853-fb7546dc2a5d" width="400">
</p>

## Problem setup and model development 

<p align="center">
  <img src="https://github.com/Berhanetek/unsupervised-detection-of-anomalous-sounds/assets/60297609/b1f881c0-92ee-48ad-85d9-fb369f3a53ab" width="600">
</p>


Is this a 2 class classification problem(Normal vs Anomalous)? Not really. This is because our training data is composed of only normal samples. So it's better to find another learning / training mechanism to learn the underlying features of the data. Used self-suervised learning for this.

Pretext task = Classification - but classification of the samples into the machine types (machine 1, machine 2) and not normal/anomalous,
However, my primary interest was in detecting anomalies, not in accurate class predictions. To achieve this, I devised a novel method of calculating anomaly scores based on the model's softmax probabilities. This allowed me to quantify the uncertainty or 'anomalousness' of each sample.

Can this be considered as self supervised?
 - No Explicit Anomaly Labels: Instead of having explicit anomaly labels, you are leveraging the model's generated labels to calculate anomaly scores. The anomaly scores serve as a measure of how anomalous a given sample is based on the model's classification certainty.
 - Utilizing Model's Representations: The classification model is learning to extract meaningful representations from the data to perform the pretext task. These representations are then repurposed to serve the anomaly detection task.
 - Self-Generated Supervision: The model learns from the patterns inherent in the data itself, essentially creating its own supervision through the classification task.
