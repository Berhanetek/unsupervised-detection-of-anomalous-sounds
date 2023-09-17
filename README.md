# Unsupervised detection of anomalous sounds

The primary objective of this project was to develop techniques for automatically identifying whether the sounds produced by industrial machines are normal or anomalous (faulty machines). This is crucial for ensuring efficient and safe operations in the context of AI-based factory automation.
The main challenge of this task is to detect unknown anomalous sounds under the condition that only normal sound samples have been provided as training data.


### Data and Preprocessing
The dataset used is taken from [Malfunctioning Industrial Machine Investigation And Inspection](https://arxiv.org/abs/1909.09347) aka MIMII dataset (Detection and Classification of Acoustic Scenes and Events 2019 Workshop)

<p align="center">
  <img src="https://github.com/Berhanetek/unsupervised-detection-of-anomalous-sounds/assets/60297609/34c826a4-a2e1-43e8-a107-228337c16f20" width="600">
</p>



Preprocessing the gived audio files is a crucial step before diving deep into any modeling. Using the raw audio files comes with some problems some of which are:
Temporal Invariance: Models trained directly on raw waveforms may struggle with variations in timing and duration. The same sound might occur at different points in time, and the model should recognize it regardless of when it happens.
High Dimensionality: Raw audio waveforms are continuous signals with high temporal resolution. They consist of amplitude values sampled at very fine time intervals. This high-dimensional data can be computationally intensive to process directly

The approach we took is to convert the raw audio files and convert them into spectrograms so we can go ahead with the problem as if it's a vision problem.

Why Spectrograms?
- Feature Extraction: Spectrograms provide a more compact and structured representation of audio signals. Instead of working with raw audio waveforms, which can be complex and difficult to analyze directly, spectrograms capture the frequency content of audio over time. This representation is often easier for models to process.
- Feature Extraction: Spectrograms inherently capture important features such as pitch, rhythm, and timbre, which are crucial for audio understanding. Deep learning models can learn these features more effectively from spectrograms
- Noise Filtering: Spectrograms can visually highlight patterns corresponding to the actual audio content while dampening the impact of constant or repetitive noise, as it appears as consistent energy across frequencies. This can make it easier for machine learning models to focus on the relevant audio features

To accomplish this we used librosa, an audio processing package

<p align="center">
  <img src="https://github.com/Berhanetek/unsupervised-detection-of-anomalous-sounds/assets/60297609/f7fd5107-b485-4035-8e62-e1dbb479805b" width="600">
</p>


