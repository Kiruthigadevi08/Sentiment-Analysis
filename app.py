from flask import Flask, render_template, request
from model import predict_sentiment

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    color = None
    emoji = None

    if request.method == 'POST':
        text = request.form['review']
        result = predict_sentiment(text)

        if result == "Positive":
            color = "positive"
            emoji = "😊"
        elif result == "Negative":
            color = "negative"
            emoji = "😠"
        else:
            color = "neutral"
            emoji = "😐"

    return render_template(
        "index.html",
        result=result,
        color=color,
        emoji=emoji
    )

if __name__ == "__main__":
    app.run(debug=True)
