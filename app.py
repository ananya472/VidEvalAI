from flask import Flask, render_template, request
import os
from pydub import AudioSegment
import whisper
from sentence_transformers import SentenceTransformer, util
import language_tool_python
from transformers import pipeline
from collections import Counter
import torch
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models once
whisper_model = whisper.load_model("base")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
tool = language_tool_python.LanguageTool('en-US')
coherence_classifier = pipeline("text-classification", model="bert-base-uncased")
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        ideal_answer = request.form.get("ideal_answer")
        video = request.files.get("video")

        if not video or not ideal_answer:
            return render_template("index.html", error="Please provide both the ideal answer and the video.")

        # Save the uploaded video
        video_filename = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
        video.save(video_filename)

        try:
            # Convert video to audio
            audio_path = video_filename.replace(".mp4", ".wav")
            video_audio = AudioSegment.from_file(video_filename, format="mp4")
            video_audio.export(audio_path, format="wav")

            # Transcribe
            result = whisper_model.transcribe(audio_path)
            transcript = result["text"]

            # Grammar
            matches = tool.check(transcript)
            grammar_score = max(0, 100 - len(matches) * 2)

            # Coherence
            raw_coherence = coherence_classifier(transcript[:512])[0]
            coherence_label_map = {
                "LABEL_0": "Possibly Off-topic",
                "LABEL_1": "Relevant and Coherent"
            }
            coherence_label = coherence_label_map.get(raw_coherence['label'], raw_coherence['label'])
            coherence_score = round(raw_coherence['score'] * 100, 2)

            # Semantic Similarity
            input_embedding = bert_model.encode(transcript, convert_to_tensor=True)
            ideal_embedding = bert_model.encode(ideal_answer, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(input_embedding, ideal_embedding).item()
            explanation_score = round(similarity * 10, 2)

            # Emotion
            chunk_length = 200
            words = transcript.split()
            chunks = [' '.join(words[i:i + chunk_length]) for i in range(0, len(words), chunk_length)]
            emotion_labels = []
            emotion_scores = []

            for chunk in chunks:
                if chunk.strip():
                    res = emotion_classifier(chunk)[0]
                    emotion_labels.append(res['label'])
                    emotion_scores.append((res['label'], res['score']))

            majority_emotion = Counter(emotion_labels).most_common(1)[0][0] if emotion_labels else "Not Detected"

            return render_template("result.html",
                transcript=transcript,
                grammar_score=grammar_score,
                coherence_label=coherence_label,
                coherence_score=coherence_score,
                explanation_score=explanation_score,
                emotion=majority_emotion
            )

        except Exception as e:
            return render_template("index.html", error=f"Processing error: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
