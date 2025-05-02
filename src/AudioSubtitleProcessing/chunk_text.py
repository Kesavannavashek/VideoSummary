import spacy

nlp = spacy.load("en_core_web_sm")

def chunk_text(input_text):
    doc = nlp(input_text)

    chunks = list(doc.sents)

    return [chunk.text for chunk in chunks]

