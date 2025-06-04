import whisper
from pydub import AudioSegment

def preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    clean_path = "cleaned_audio.wav"
    audio.export(clean_path, format="wav")
    return clean_path

model = whisper.load_model("small")

def transcribe_audio(file_path):
    clean_file = preprocess_audio(file_path)

    result = model.transcribe(clean_file)
    return {
        "language_code": result["language"],
        "transcription": result["text"]
    }

# Run it
if __name__ == "__main__":
    input_file = "hindi_sample.mp3"
    output = transcribe_audio(input_file)
    print("Detected Language:", output["language_code"])
    print("Transcription:", output["transcription"])
