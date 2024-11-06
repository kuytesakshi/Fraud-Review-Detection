from flask import Flask , render_template , request
import pickle
from tensorflow.keras.models import load_model
from prediction import predict

model = load_model("LSTM.h5")

tokenizer = pickle.load(open("Data\\tokenized\\tokenizer.pkl", "rb"))


app = Flask(__name__)

@app.route("/")
def main_page():
    return render_template("index.html")

@app.route("/submit" , methods = ["POST"])
def sub():
    text = request.form.get("review")
    result = ""
    if text == "":
        result = "No review entered..."
    else:
        result = predict(text , model=model , tokenizer=tokenizer)
    return render_template("index.html" ,  result = result)


if __name__ == "__main__":
    app.run(debug = True)