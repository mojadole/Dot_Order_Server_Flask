import flask
import os
from flask import Flask,  jsonify, request
import main


app = flask.Flask(__name__)


@app.route('/')
def hello_world():
    return '아니 apple 5000포트 뭔데;;;;;'

@app.route('/search_menu', methods=['GET'])
def search_menu():
    menu_name = request.args.get('menu_name')  # Extract menu_name from query parameters
    if not menu_name:  # If menu_name is not provided
        return jsonify({"error": "Missing menu_name query parameter"}), 400  # Return an error message
    results = main.main_search_menu(menu_name)  # Call the function in main.py
    return jsonify(results)  # Return the results as JSON


if __name__ == "__main__":
    app.run(host='0.0.0.0', port ='4444')