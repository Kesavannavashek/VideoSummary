import spacy

# Load the spaCy model (You can use 'en_core_web_sm' for a lighter model)
nlp = spacy.load("en_core_web_sm")


def chunk_text(input_text):
    # Process the text using spaCy
    doc = nlp(input_text)

    # Break the text into sentences (chunks)
    chunks = list(doc.sents)

    # Return the list of chunks
    return [chunk.text for chunk in chunks]

