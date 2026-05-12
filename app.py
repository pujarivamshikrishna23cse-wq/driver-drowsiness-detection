from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Driver Drowsiness Detection Running Successfully"

if __name__ == "__main__":
    app.run()