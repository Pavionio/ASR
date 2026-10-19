# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a solution of ASR homework with PyTorch. It is based on Conformer-CTC model with RoPE, a BPE tokenizer and a beam search with LM shallow fusion.

See the task assignment [here](https://github.com/markovka17/dla/tree/2026/hw1ASR).

See the report [here](https://www.comet.com/qwertypavel491/dla-asr/reports/l6dQQM0nugNRDBQyn2CkzDU6q).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n dla_asr python=3.12

   # activate env
   conda activate dla_asr
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/3.12/bin/python3 -m venv dla_asr

   # alternatively, using default python version
   python3 -m venv dla_asr

   # activate env
   source dla_asr/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```
   For GPU execution, install a CUDA-compatible PyTorch build.


2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

3. Download the trained model, BPE tokenizer and language model:
   ```bash
   python -m scripts.download_resources
   ```


## Training

Training uses Hydra configs from `src/configs`. Configure Comet ML before running:

```bat
set "COMET_API_KEY=YOUR_API_KEY"
```

The command above is for Windows Anaconda Prompt. On Linux, use:

```bash
export COMET_API_KEY=YOUR_API_KEY
```

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

The final model is trained in two stages:

```bash
python train.py -cn train_clean_460
python train.py -cn train_full_960
```

The first stage uses train-clean-100 and train-clean-360. The second stage initializes from the first stage's best weights and trains on all 960 hours.

To overwrite an existing experiment directory, add `trainer.override=true`.


## Inference

Place `.wav`, `.flac` or `.mp3` files in the following directory:

```text
data/custom/audio/
├── example1.wav
└── example2.flac
```

Run inference:

```bash
python inference.py
```

Predictions are saved as matching text files:

```text
data/predictions/test/
├── example1.txt
└── example2.txt
```
Paths can be overridden:

```bash
python inference.py datasets.test.data_dir=path/to/dataset inferencer.save_path=path/to/predictions
```

The dataset directory must contain an `audio` subdirectory. `transcriptions` subdirectory is optional and used only for evaluation. Predictions are saved in the `test` subdirectory of `inferencer.save_path`.

## Evaluation

Place reference transcriptions in `data/custom/transcriptions` (or override --target-dir parameter). Each reference must have the same filename stem as its audio file:

```text
data/custom/transcriptions/
├── example1.txt
└── example2.txt
```

After inference, calculate corpus CER and WER:

```bash
python -m scripts.calc_metrics --target-dir data/custom/transcriptions --pred-dir data/predictions/test
```

Here, `--target-dir` points to the reference transcriptions and `--pred-dir` points to the predicted transcriptions. If you changed `inferencer.save_path`, set `--pred-dir` to its `test` subdirectory.

Final LibriSpeech test results:

| Dataset | CER | WER |
|---|---:|---:|
| test-clean | 1.97% | 5.58% |
| test-other | 6.91% | 14.50% |


## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
