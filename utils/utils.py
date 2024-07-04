from utils.variables import chatwords
import re
import string


def remove_html(text):
    """
    Removes HTML tags from the input text.

    Parameters:
        text (str): The input text from which HTML tags will be removed.

    Returns:
        str: The input text with HTML tags removed.
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def remove_url(text):
    """
    Removes URLs from the input text.

    Parameters:
        text (str): The input text from which URLs will be removed.

    Returns:
        str: The input text with all URLs removed.
    """
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_punctuations(text):
    """
    Removes punctuation from the given text.

    Parameters:
        text (str): The input text from which punctuation will be removed.

    Returns:
        str: The input text with all punctuation removed.
    """
    punctuations = string.punctuation
    no_punctuations = "".join(
        [char for char in text if char not in punctuations])
    return no_punctuations


def chat_conversation(text):
    """
    Replaces chat words in the given text with their corresponding replacements from the chatwords dictionary.

    Parameters:
        text (str): The input text to process.

    Returns:
        str: The processed text with chat words replaced.
    """
    new_text = []
    for word in text.split():
        if word.upper() in chatwords:
            new_text.append(chatwords[word.upper()])
        else:
            new_text.append(word)
    return " ".join(new_text)

# Preprocess user input


def preprocess(text):
    """
    Preprocesses the given text by performing the following steps:
    1. Converts the text to lowercase.
    2. Removes HTML tags from the text using the `remove_html` function.
    3. Removes URLs from the text using the `remove_url` function.
    4. Removes punctuations from the text using the `remove_punctuations` function.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    text = chat_conversation(text)
    text = text.lower()
    text = remove_html(text)
    text = remove_url(text)
    text = remove_punctuations(text)
    return text
