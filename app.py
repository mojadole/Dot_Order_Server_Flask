import flask

app = flask.Flask(__name__)


@app.route('/')
def hello_world():
    return '아니 apple 5000포트 뭔데;;;;;'


if __name__ == "__main__":
    app.run(host='0.0.0.0')