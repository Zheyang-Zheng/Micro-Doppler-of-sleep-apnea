# Micro-Doppler-of-sleep-apnea
this page shares the MATLAB and python code of using FMCW radar to recognize sleep apnea

This repository contains the final report and code for a project studying sleep apnea detection using a 24 GHz FMCW radar. The radar collects IQ data (in-phase and quadrature signals) from participants performing both normal breathing and simulated apnea movements. Using MATLAB, the raw data is processed into time–frequency spectrograms, revealing characteristic micro-Doppler signatures. Several machine learning classifiers (MobileNetV2, ResNet, Vision Transformer, and SVM) are then evaluated on these spectrograms to determine their effectiveness in distinguishing apnea from normal breathing. Experimental results show promising accuracy for lightweight models like MobileNetV2 and SVM, whereas deeper architectures tend to overfit the small dataset.

MATLAB Scripts: Used for radar signal processing (range FFT, MTI filtering, STFT-based spectrogram generation, and optional IQ phase extraction).

Python Code: Implements classification using MobileNetV2, ResNet, ViT, and SVM on the generated spectrogram images (or phase-based data).

# MATLAB Code Usage

Below is a concise outline of how to work with the MATLAB scripts for radar data processing. These steps are summarized from the final report; please refer there for more details if needed.

    Load the Radar Data:

        Place your binary *.dat file(s) (recorded by Ancortek SDR-KIT software) in a known directory.

        Modify the data_dir and file_name variables in the MATLAB script to point to the correct file path.

    Read and Parse Header:

        The first few values in the *.dat file store radar parameters (center frequency, sweep time, number of time samples per sweep, bandwidth).

        The script automatically extracts these parameters for use in downstream processing.

    Reshape IQ Data:

        The raw IQ samples are split into two channels (I/Q), reshaped into a matrix representing [samples per chirp × number of chirps].

        If dual-channel data is included, each channel is processed separately.

    Range FFT:

        A fast-time FFT is applied to each chirp to convert beat frequencies into range bins.

        Only the positive half of the FFT is retained for practical reasons (real-valued beat signals).

    MTI (Moving Target Indicator) Filtering:

        A high-pass filter is applied along the slow-time dimension to remove stationary clutter (walls, stationary objects).

        Optional: Discard the first range bin if it contains strong direct coupling or filter artifacts.

    Select Target Range Bin and Generate Spectrogram:

        Identify the range bin corresponding to the chest location (e.g., bin 3–4).

        Perform an STFT (Short-Time Fourier Transform) over slow-time to produce a spectrogram.

        Parameters typically include a window size (e.g., 256), overlap (e.g., 90%), and FFT length.

    (Optional) IQ Phase Extraction:

        Some scripts may compute the instantaneous phase from I/Q data to examine breathing patterns directly in the time domain.

        Note that short data segments (10 s) may not yield strong apnea vs. normal distinctions in phase signals.

    Save or Export Results:

        The final output is usually saved as images (PNG/JPG) or MATLAB figures, which can be used for classification in Python.

 # Python Classification Code Usage
 Prepare Spectrogram Images:

    Run the MATLAB pipeline on each *.dat file to generate spectrogram images (e.g., *.png).

    Place these images into labeled folders (e.g., normal_breathing and apnea).

Install Dependencies:

    Python 3.x

    NumPy, Matplotlib, Pillow, scikit-learn, TensorFlow/PyTorch (depending on your classifier implementation)

    Possibly other packages like OpenCV or Pandas, if used by your scripts.

Run Classification Script:

    There may be separate Python files for each classifier (e.g., classify_mobilenetv2.py, classify_resnet.py, etc.).

    Update any relevant paths or hyperparameters (e.g., directories for training/test images, model checkpoint paths).

    Execute the script.

    Scripts typically load images, split them into train/test sets, perform data augmentation if enabled, and then train the model or run inference. Accuracy and confusion matrices should be displayed or saved as images.

# Repository Structure 

.
├── final_report.pdf
├── MATLAB_codes
│   ├── data_processing.m
│   ├── spectrogram_generation.m
│   ├── iq_phase_extraction.m
│   └── ...
├── Python_codes
│   ├── classify_mobilenetv2.py
│   ├── classify_resnet.py
│   ├── classify_vit.py
│   ├── classify_svm.py
│   └── ...
└── README.md

Save your data in the form showing below:
C:/ random_memory
│
├── training
│   ├── apnea
│   └── breathnorm
│
└── test
    ├── apnea
    └── breathnorm
