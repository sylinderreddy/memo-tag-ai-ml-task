from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import os
from pydub import AudioSegment  # ✅ Required for MP3 to WAV conversion

from memo_tag_analysis import (
    transcribe_audio_whisper,
    extract_audio_features,
    detect_hesitations,
    pause_rate,
    detect_word_recall,
    naming_task_check,
    incomplete_sentence_count
)

app = FastAPI(
    title="MemoTag Cognitive Analysis API",
    description="Upload a voice file to get cognitive speech features.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "🧠 Welcome to the MemoTag API!",
        "usage": "Use /docs or POST to /analyze-audio/ with an audio file."
    }

@app.post("/analyze-audio/")
async def analyze_audio(file: UploadFile = File(...)):
    original_path = f"temp_{file.filename}"
    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 🔄 Convert MP3 to WAV if needed
        if file.filename.lower().endswith(".mp3"):
            wav_path = original_path.replace(".mp3", ".wav")
            audio = AudioSegment.from_mp3(original_path)
            audio.export(wav_path, format="wav")
        else:
            wav_path = original_path

        # 📊 Expected term sets
        expected = ["coffee", "breakfast", "shower", "toothbrush", "bag"]
        animals = ["dog", "cat", "cow", "lion", "elephant", "goat", "tiger"]

        # 🧠 Run feature extraction
        text = transcribe_audio_whisper(wav_path)
        audio_feats = extract_audio_features(wav_path)

        features = {
            "hesitation_count": detect_hesitations(text),
            "pauses": pause_rate(wav_path),
            "tempo": audio_feats["tempo"],
            "pitch_var": audio_feats["pitch_variability"],
            "missing_recall_words": detect_word_recall(text, expected),
            "named_animals": naming_task_check(text, animals),
            "incomplete_sentences": incomplete_sentence_count(text)
        }

        return JSONResponse(content={"status": "success", "features": features})

    except Exception as e:
        print(f"❌ Internal Error: {e}")  # Logs to terminal
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

    finally:
        # 🧹 Clean up temp files
        if os.path.exists(original_path):
            os.remove(original_path)
        if os.path.exists(wav_path) and wav_path != original_path:
            os.remove(wav_path)
