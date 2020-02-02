from dataset import ANTONYM, COHYPONYM, HYPERNYM, CORRUPTION, RELATIONS, AnnotatedWord

# TODO [MASK] -> tokenizer mask token!!

WORD_TOKEN = '<W>'
MASK_TOKEN = '[MASK]'

def get_patterns(word: AnnotatedWord, relation: str):
    if relation == ANTONYM:
        return get_patterns_antonym(word)
    if relation == COHYPONYM:
        return get_patterns_cohyponym(word)
    if relation == HYPERNYM:
        return get_patterns_hypernym(word)
    if relation == CORRUPTION:
        return get_patterns_corruption(word)
    raise ValueError("No patterns found for relation {}".format(relation))


def get_patterns_antonym(_):
    return [
        '<W> is the opposite of [MASK]',
        '<W> is not [MASK]',
        'someone who is <W> is not [MASK]',
        'something that is <W> is not [MASK]',
        '" <W> " is the opposite of " [MASK] "'
    ]


def get_patterns_hypernym(word):
    article = _get_article(word)

    return [
        '<W> is a [MASK]',
        '<W> is an [MASK]',
        article + ' <W> is a [MASK]',
        article + ' <W> is an [MASK]',
        '" <W> " refers to a [MASK]',
        '" <W> " refers to an [MASK]',
        '<W> is a kind of [MASK]',
        article + ' <W> is a kind of [MASK]'
    ]


def get_patterns_cohyponym(_):
    return [
        '<W> and [MASK]',
        '" <W> " and " [MASK] "'
    ]


def get_patterns_corruption(_):
    return [
        '" <W> " is a misspelling of " [MASK] " .',
        '" <W> " . did you mean " [MASK] " ?'
    ]


def _get_article(word):
    if word.word[0] in ['a', 'e', 'i', 'o', 'u']:
        return 'an'
    return 'a'


if __name__ == '__main__':

    dummy = AnnotatedWord('dummy', pos='n', freq=1, count=1)

    for rel in RELATIONS:
        try:
            print('=== {} patterns ({}) ==='.format(rel, len(get_patterns(dummy, rel))))
            for p in get_patterns(dummy, rel):
                print(p)
        except ValueError:
            print('=== no patterns found for relation {} ==='.format(rel))
        print('')
