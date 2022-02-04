from lib.officeChatbot import *
from flask import Flask, jsonify, request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

characters_to_awnser = ['Michael','Dwight','Jim','Pam','Andy','Erin','Kevin',
 'Angela','Oscar','Darryl','Ryan','Phyllis','Nellie','Toby',
 'Kelly','Stanley','Meredith','Robert','Holly','Gabe']

df = pd.read_csv('./the-office_lines.csv')
X = sparse.load_npz("./TfId.npz")
vectorizer = joblib.load("./Vectorizer")

chatbot = similaritySentenceDetector()

@app.route('/api/init', methods=['GET'])
def init():
    if(request.method == 'GET'):
        data = {"data": "test response"}
        return jsonify(data)

@app.route('/api/talk', methods = ['POST'])
def index():
    user_input = request.json
    sentence_input = user_input['user_input']
    character, response, similarity = chatbot.get_response(sentence_input, df, X, vectorizer, characters_to_awnser)

    return jsonify({
        "character" : str(character),
        "response" : str(response),
        "similarity" : str(similarity)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)