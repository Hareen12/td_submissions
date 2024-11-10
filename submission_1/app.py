# This should be replaced with your starter code!
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request
from submission_1 import starter_code_model


# Please do NOT modify this file
# Modifying this file may cause your submission to not be graded

app = Flask(__name__)
# @app.post("/")
@app.route("/", methods=["POST"])
def challengeSetup():
	req_data = request.get_json()
	words = req_data['words']
	strikes = req_data['strikes']
	isOneAway = req_data['isOneAway']
	correctGroups = req_data['correctGroups']
	previousGuesses = req_data['previousGuesses']
	error = req_data['error']

	guess, endTurn = starter_code_model.model(words, strikes, isOneAway, correctGroups, previousGuesses, error)

	return {"guess": guess, "endTurn": endTurn}

if __name__ == '__main__':
    app.run(debug=True, port=5000)
