import librosa
import numpy as np
import whisper

# Load Whisper model once (for reuse)
model = whisper.load_model("base", device="cpu")

# 1. Transcribe audio to text
def transcribe_audio_whisper(file_path):
    result = model.transcribe(file_path)
    return result['text']

# 2. Extract acoustic features
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_variability = np.std(pitches[pitches > 0])
    return {
        "duration": duration,
        "tempo": tempo,
        "pitch_variability": pitch_variability
    }

# 3. Count hesitation words
def detect_hesitations(text):
    hesitations = ['uh', 'um', 'erm', 'eh']
    words = text.lower().split()
    count = sum(words.count(h) for h in hesitations)
    return count

# 4. Count number of pauses
def pause_rate(file_path):
    y, sr = librosa.load(file_path, sr=None)
    intervals = librosa.effects.split(y, top_db=20)
    pauses = len(intervals) - 1
    return pauses

# 5. Word recall (missing expected words)
def detect_word_recall(text, expected_words):
    words_in_text = set(text.lower().split())
    missing = [word for word in expected_words if word not in words_in_text]
    return len(missing)

# 6. Naming task (e.g., animals)
def naming_task_check(text, category_keywords):
    words_in_text = set(text.lower().split())
    matched = [word for word in words_in_text if word in category_keywords]
    return len(matched)

# 7. Incomplete sentence detection
def incomplete_sentence_count(text):
    sentences = text.strip().split(".")
    incomplete = [s for s in sentences if len(s.strip().split()) < 3]
    return len(incomplete)
