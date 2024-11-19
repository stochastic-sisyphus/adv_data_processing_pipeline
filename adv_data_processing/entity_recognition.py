import spacy
from typing import List, Tuple
import dask.dataframe as dd
import logging

logger = logging.getLogger(__name__)

def extract_entities(texts: dd.Series) -> List[Tuple[str, str]]:
    """Extract named entities from a series of texts."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spacy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Convert dask series to pandas for processing
    texts = texts.compute()
    
    entities = []
    for text in texts:
        doc = nlp(str(text))
        entities.extend([(ent.text, ent.label_) for ent in doc.ents])
    
    return entities

