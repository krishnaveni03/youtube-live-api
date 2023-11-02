from flask import Flask, render_template, request, redirect, url_for
import re
import googleapiclient.discovery
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load your trained sentiment analysis model
model = keras.models.load_model('sentiment_analysis.h5')

# Tokenizer configuration (must match the one used for training)
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)

# Initialize the YouTube Data API client
DEVELOPER_KEY = 'AIzaSyDub1h7J9kgxhRTZaWHi7HH-3Nr5DMzWYA'  # Replace with your YouTube API key
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEVELOPER_KEY)

def get_youtube_comments(video_id, max_comments=10):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_comments
    )
    response = request.execute()
    for comment in response["items"]:
        comments.append(comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
    return comments

#@tf.function
def predict_sentiment(comment, model, tokenizer, threshold=0.5):
    tokenizer.fit_on_texts([comment])
    sequences = tokenizer.texts_to_sequences([comment])
    X = pad_sequences(sequences, maxlen=100)
    prediction = model(X)
    sentiment = "positive" if prediction[0][0] > threshold else "negative"
    return sentiment


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_link = request.form['video_link']
        comment_index = int(request.form['comment_index'])
        video_id_match = re.search(r"(?<=v=)[^&]+", video_link)
        if video_id_match:
            video_id = video_id_match.group(0)
        else:
            return "Invalid YouTube video link format."
        comments = get_youtube_comments(video_id, max_comments=10)
        if 0 <= comment_index < len(comments):
            sentiment = predict_sentiment(comments[comment_index], model, tokenizer)
            return render_template('result.html', comment=comments[comment_index], sentiment=sentiment)
        else:
            return "Invalid comment index."

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)