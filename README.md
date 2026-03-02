# Speaker-Labeled Transcription (WhisperX + pyannote)

This project generates:
- transcript text
- speaker labels (`SPEAKER_00`, `SPEAKER_01`, ...)

It uses:
- `WhisperX` for speech-to-text + alignment
- `pyannote` for speaker diarization

## Output Files

For input `audio/my_file.mp3`, output will be:
- `audio/my_file_speakers.txt` (readable transcript)
- `audio/my_file_speakers.json` (full structured data)

Example:

```txt
[00:00:01.240 - 00:00:04.920] SPEAKER_00: Hi everyone, thanks for joining.
[00:00:05.100 - 00:00:08.410] SPEAKER_01: Sure, let's start with the update.
```

## One-Time Setup

### 1) Install requirements

- Python 3.9+ (3.11 recommended)
- `ffmpeg`
- Hugging Face account

### 2) Create virtual environment and install dependencies

```bash
cd /path/to/vercel-sdk-ai-chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Hugging Face token and model access (one-time)

Create one token with:
- Permission: `Read`

Accept terms on these model pages while logged into the same account:
- `https://huggingface.co/pyannote/segmentation-3.0`
- `https://huggingface.co/pyannote/speaker-diarization-3.1`

## Every Time You Open a New Terminal

Run these two commands again:

```bash
cd /path/to/vercel-sdk-ai-chatbot
source .venv/bin/activate
export HF_TOKEN="hf_your_token_here"
```

## Run On a New Audio File

### 1) Put file in `audio/` folder

Example:
- `audio/new_meeting.mp3`

### 2) Run with `small` model (faster)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 transcribe_with_speakers.py audio/meeting.mp3 \
  --model small \
  --language en \
  --device cpu \
  --compute-type int8 \
  --batch-size 1 \
  --min-speakers 2 \
  --max-speakers 2
```

### 3) Run with `medium` model (better quality, slower)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 transcribe_with_speakers.py audio/new_meeting.mp3 \
  --model medium \
  --language en \
  --device cpu \
  --compute-type int8 \
  --batch-size 1 \
  --min-speakers 2 \
  --max-speakers 2
```

## Optional: Save with custom output name

```bash
python3 transcribe_with_speakers.py audio/new_meeting.mp3 \
  --model small \
  --language en \
  --device cpu \
  --compute-type int8 \
  --batch-size 1 \
  --output audio/new_meeting_small
```

This creates:
- `audio/new_meeting_small.txt`
- `audio/new_meeting_small.json`

## Troubleshooting

If you see NumPy compatibility errors:

```bash
pip install -r requirements.txt --upgrade --force-reinstall
```

