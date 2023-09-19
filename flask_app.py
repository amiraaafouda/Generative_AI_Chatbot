
from flask import Flask, request, jsonify

# Import the GPT2 class
# from gpt2 import GPT2
from main import GPT2
app = Flask(__name__)

# Create an instance of GPT2
gpt2 = GPT2(model_name="1558M")

# Define a route for generating text
@app.route('/generate', methods=['POST'])
def generate_text():
    # Get the input text from the request
    raw_text = request.json['text']

    # Generate conditional text using GPT-2
    result = gpt2.generate_conditional(raw_text)

    # Return the generated text as a JSON response
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run()