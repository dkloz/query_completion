def ids_to_str(ids, inv_vocab):
    """Converts a list of ids to a sentence that can be read by humans.
    Args:
        ids: List of ids (corresponding to words).
        inv_vocab: Inverted vocab dictionary. From word_id -> word.
    Returns:
        string of sentence.
    """
    words = []
    for word_id in ids:
        if word_id > 0:
            words.append(inv_vocab.get(word_id - 1, '-'))
    return ''.join(words)

