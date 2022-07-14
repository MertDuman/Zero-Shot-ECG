# A Personalized Zero-Shot ECG Arrhythmia Monitoring System

### [Paper](-----)

## Setup

Python (3.8.10) dependencies:

* matplotlib (3.4.2), numpy (1.19.5), scipy (1.6.3), pandas (1.2.4), seaborn (0.11.1, optional)
* torch (1.10.2+cu113)
* [wfdb](https://pypi.org/project/wfdb/) (3.3.0)
* [import_ipynb](https://pypi.org/project/import-ipynb/) (0.1.3)

To minimize conflict, our versions are given as reference.

## Replicating Our Results

1. Download the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).
2. Run ecg_beat_extraction.ipynb twice to generate single beats and beat-trios.
    - There should be 6 files generated.
3. Run ecg_dataset_preparation.ipynb twice to generate datasets from single beats and beat-trios.
    - This generates dictionaries for each user. Save the dictionaries for single beats (and optionally for beat-trios).
    - We perform domain adaptation at this stage using the generated dictionaries for each user.
    - There should be 68 dataset files generated, and dictionaries for each user.
4. To train your own classifier, go to train.ipynb. Otherwise, skip this part.
5. To test with pretrained weights go to pretrained_*.ipynb files.
