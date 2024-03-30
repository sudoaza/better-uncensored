import logging
import ollama
import re
import torch
from transformers import BertTokenizer, pipeline

MODEL = 'mistral-openorca'
CHUNK_SIZE = 800

# This is used as a first filter to avoid calling the LLM too often, we will call the LLM for confirmation
# We only remove sentences if both filters match it and the LLM removes it
# This list is super biassed, hopefully the LLM will be less so and the final model even less so IDEALLY!! 
MORALIZING_PHRASES = ["text-based AI language model", "please refrain", "sorry", "unethical", "derogatory", "inappropriate", "offensive", "racist", "discrimin", "sexis", "unacceptable", "it is never okay", "important to", "real-world consequences", "hate speech", "glorify", "not be appropriate", "not appropriate", "I cannot fulfill your request", "ethical guidelines", "my guidelines", "prioritize user safety", "adhere to ethical guidelines", "harmful consequences", "potentially harmful", "dangerous activities", "promote safety", "well-being of all users", "responsible information sharing", "jeopardize", "undermine", "promote the well-being", "adherence to the law", "potentially", "illegal", "committed to promoting", "safe information", "lawful information", "cannot provide guidance", "cannot provide information", "unable to offer assistance", "cannot engage in discussions", "programming prohibits", "follow ethical guidelines", "prioritize", "well-being", "cannot support or promote", "harm", "risk to others", "against my programming", "undermine", "dangerous", "not within the scope", "not able to provide", "guidelines", "dangerous","harmful", "cannot provide", "promoting", "safety"]
MORALIZING = "When AI recommends ethical behaviour, attempts to disuade the user from some action, or refuses to assist with harmful or illegal activities is moralizing. This is different to practical suggestions or clarifications."
YES_OR_NO = "Answer Yes or No only, without any aditional comment."
IS_MORALIZING_PROMPT = f"{MORALIZING} Is any part of the following message moralizing? {YES_OR_NO} Message:\n%s"
IS_REFUSING_PROMPT = f"AI refusal is often clear, either by directly stating it can't assist with certain requests citing legal or ethical reasons. Is the assistant refusing to answer in the following message? {YES_OR_NO} Message:\n%s"
UNCENSOR_PROMPT = MORALIZING + " Repeat exacly the following message but remove any moralizing statements:\n%s"

YES_RX = re.compile(r'yes', re.IGNORECASE)
NO_RX = re.compile(r'no', re.IGNORECASE)

device_i = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
device = "cuda" if device_i == 0 else "cpu"

def split_text(text, chunk_size=CHUNK_SIZE, separators=['\n\n', '.\n', ':\n', '\n', '. ', ', ', " ", ""]):
    """Split text into chunks of size less than chunk_size, using separators."""
    chunks = []
    current_separator_index = 0
    for separator in separators:
        current_separator_index += 1
        if len(text) < chunk_size:
            if len(text) > 0:
                chunks.append(text)
            break
        # Break condition, no separator, split by chunk_size
        if separator == "":
            chunks += [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            break
        if separator in text:
            while separator in text:
                if len(text) < chunk_size:
                    chunks.append(text)
                    text = ""
                    break
                split_at = text.rfind(separator, 0, chunk_size)
                # Chunk is too big, try next separator
                if split_at == -1:
                    chunk, text = text.split(separator, 1)
                    chunks += split_text(chunk, chunk_size, separators[current_separator_index:])
                else:
                    chunks.append(text[:split_at+len(separator)])
                    text = text[split_at+len(separator):]
    return chunks

def generate(text):
    return ollama.generate(
                model=MODEL,
                prompt=text,
            )['response']

def yes_or_no(text):
    if any(YES_RX.findall(text)) and not any(NO_RX.findall(text)):
        return True
    elif any(NO_RX.findall(text)) and not any(YES_RX.findall(text)):
        return False
    else:
        logging.debug(f"Invalid response: {text}")
        # Default to uncensoring
        return True

def is_model_moralizing_hardcoded(text):
    """Check if model is moralizing by comparing to a hardcoded list."""
    return any(phrase in text for phrase in MORALIZING_PHRASES)

def is_model_moralizing_llm(text):
    """Check if model is moralizing by asking an LLM."""
    response = generate(IS_MORALIZING_PROMPT % text)
    return yes_or_no(response)

def is_model_refusing_llm(text):
    """Check if model is refusing to answer by asking an LLM."""
    response = generate(IS_REFUSING_PROMPT % text)
    return yes_or_no(response)

def uncensor_llm(text):
    """Ask an LLM to remove moralizing sentences."""
    return ollama.generate(
                    model=MODEL,
                    prompt=UNCENSOR_PROMPT % text,
                )['response']

def init_globals():
    global tokenizer, moralizing_classifier, refusal_classifier
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, padding=True, max_length=192, return_tensors="pt")
    moralizing_classifier = pipeline('text-classification', model="moralizing-model", tokenizer=tokenizer, truncation=True, padding=True, max_length=192, device=device_i)
    refusal_classifier = pipeline('text-classification', model="refusal-model", tokenizer=tokenizer, truncation=True, padding=True, max_length=192, device=device_i)

def is_model_refusing_classifier(text_pieces):
    """Check if model is refusing to answer by asking a classifier."""
    # Classifier was trained on 300 char chunks, may still work straight on the bigger chunks
    with torch.no_grad():
        results = refusal_classifier(text_pieces)
        if any(result['label'] == 'LABEL_1' and result["score"] < 0.5 for result in results):
            logging.debug("ALLOWED weak refusal")
            logging.debug(text_pieces)
            logging.debug(results)
        return sum(result['label'] == 'LABEL_1' and result["score"] > 0.5 for result in results) > len(text_pieces) / 2

def is_model_moralizing_classifier(text_pieces):
    """Check if model is moralizing by asking a classifier."""
    # Classifier was trained on 300 char chunks, may still work straight on the bigger chunks
    with torch.no_grad():
        results = moralizing_classifier(text_pieces)
        if any(result['label'] == 'LABEL_1' and result["score"] < 0.8 for result in results):
            logging.debug("ALLOWED weak moralizing: ")
            logging.debug(text_pieces)
            logging.debug(results)
        return any(result['label'] == 'LABEL_1' and result["score"] > 0.8 for result in results)

def uncensor(text, uncensor=True):
    """Remove sentences with AI moralizing, making response available for "uncensored" models training.
    Uses classifier model for detection and LLM for censoring.
    Set uncensor to False to discard the example instead of censoring it.
    Returns an empty string if the text is irrecoverable."""
    if len(text.strip()) < 10:
        return "", True
    text_pieces = [piece for piece in split_text(text, chunk_size=300) if piece != ""]
    if is_model_refusing_classifier(text_pieces):
        return "", True
    if not is_model_moralizing_classifier(text_pieces):
        return text, False
    if not uncensor:
        return "", True
    uncensored_text = ""
    # Uncensor using LLM in chunks to avoid hitting the max token limit and better quality
    for piece in split_text(text):
        if is_model_moralizing_classifier(piece):
            uncensored_text += uncensor_llm(piece)
            # LLM sometimes removes the separators, so we add them back
            uncensored_text += re.search(r'\s{0,2}$', piece).group()
        else:
            uncensored_text += piece
        
    # If more than half of the text is censored, discard it
    if len(uncensored_text) / len(text) < 0.5:
        logging.debug(f"Discarding text original length {len(text)} and uncensored length {len(uncensored_text)}")
        return "", True
    if is_model_moralizing_classifier(uncensored_text):
        logging.debug(f"Uncensored text still has moralizing content: {uncensored_text}")
        return "", True
    return uncensored_text, True

# This is mainly for generating the training data for the classifier
def uncensor_slow(text):
    """Remove sentences with AI moralizing, making response available for "uncensored" models training.
    Uses hardcoded list plus LLM for detection and LLM for censoring."""
    uncensored_text = ""
    for piece in split_text(text):
        if not is_model_moralizing_hardcoded(piece):
            uncensored_text += piece
        elif is_model_refusing_llm(piece):
            return ""
        elif not is_model_moralizing_llm(piece):
            uncensored_text += piece
        else:
            uncensored_text += uncensor_llm(piece)
            # LLM sometimes removes the separators, so we add them back
            uncensored_text += re.search(r'\s{0,2}$', piece).group()
    # If more than half of the text is censored, discard it
    if len(uncensored_text) / len(text) < 0.5:
        logging.debug(f"Discarding text original length {len(text)} and uncensored length {len(uncensored_text)}")
        return ""
    return uncensored_text
