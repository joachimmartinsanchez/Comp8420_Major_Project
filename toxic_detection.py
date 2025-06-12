from detoxify import Detoxify

# Initialize Detoxify once
tox_model = Detoxify('original')

def is_toxic(text):
    # Run toxicity check
    tox_scores = tox_model.predict(text)
    is_toxic = tox_scores["toxicity"] > 0.5

    return (is_toxic)
