# MemoTag Voice Cognitive Analysis - AI/ML Task

This project is a proof-of-concept voice analysis pipeline that extracts speech-based cognitive features and identifies potential indicators of cognitive decline.

## Features Extracted

- Hesitation markers (`uh`, `um`)
- Pauses per sentence
- Tempo (speech rate)
- Pitch variability
- Word recall (missing key words)
- Word association (e.g., naming animals)
- Sentence completeness

##  Project Structure

```bash
memo_tag_project/
├── memo_tag_analysis.py     # core feature extractor
├── api_server.py            # FastAPI endpoint
├── run_locally.py           # local testing script
├── requirements.txt         # install this before running
└── audio_samples/           # optional voice clips
