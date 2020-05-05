from flask import Flask, request, Response, json


app = Flask(__name__)

@app.route('/api/', methods=['GET', 'POST'])
@app.route('/api', methods=['GET', 'POST'])
def predict():
    data = request.get_json(force=True)
    requestData = [data['...']]
    #print("hello world")


if __name__ == '__main__':
    app.run()