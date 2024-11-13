# GitHub Action: Download and Process MP3

This repository contains a GitHub Action workflow to download an MP3 file from a given YouTube URL and process it with the Whisper JAX package.

## Usage

To use this GitHub Action workflow, follow these steps:

1. Create a new file in your repository at `.github/workflows/download_and_process.yml`.
2. Copy the following content into the file:

```yaml
name: Download and Process MP3

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  download-and-process:
    runs-on: ubuntu-latest

    env:
      PYTHONPATH: /home/junfan/test/venv/lib/python3.10

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install youtube-dl whisper-jax

    - name: Download MP3 from YouTube
      run: youtube-dl -x --audio-format mp3 <YOUTUBE_URL>

    - name: Process MP3 with Whisper JAX
      run: |
        from whisper_jax import FlaxWhisperPipline
        pipeline = FlaxWhisperPipline("openai/whisper-large-v2")
        text = pipeline("audio.mp3")
        echo $text
```

3. Replace `<YOUTUBE_URL>` with the actual YouTube URL you want to download the MP3 from.

4. Commit and push the changes to your repository.

## Dependencies

This workflow requires the following dependencies:

- `youtube-dl`: A command-line program to download videos from YouTube and other video sites.
- `whisper-jax`: A package for processing audio files with the Whisper JAX pipeline.

Make sure to install these dependencies in your Python environment.
