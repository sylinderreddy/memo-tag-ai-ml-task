import sys, os
from memo_tag_analysis import (
    transcribe_audio_whisper,
    extract_audio_features,
    detect_hesitations,
    pause_rate,
    detect_word_recall,
    naming_task_check,
    incomplete_sentence_count
)

def analyze_file(audio_path):
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return

    expected = ["coffee", "breakfast", "shower", "toothbrush", "bag"]
    animals = ["dog", "cat", "cow", "lion", "elephant", "goat", "tiger"]

    print(f"🔍 Analyzing {audio_path}...")

    try:
        text = transcribe_audio_whisper(audio_path)
        audio_feats = extract_audio_features(audio_path)

        features = {
            "hesitation_count": detect_hesitations(text),
            "pauses": pause_rate(audio_path),
            "tempo": audio_feats["tempo"],
            "pitch_var": audio_feats["pitch_variability"],
            "missing_recall_words": detect_word_recall(text, expected),
            "named_animals": naming_task_check(text, animals),
            "incomplete_sentences": incomplete_sentence_count(text)
        }

        print("\n🧠 Cognitive Speech Features:")
        for key, value in features.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_locally.py <audiofile.wav or .mp3>")
    else:
        analyze_file(sys.argv[1])
