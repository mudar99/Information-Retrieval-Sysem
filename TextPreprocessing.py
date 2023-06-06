import contractions
from autocorrect import Speller
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
# punctuations = string.punctuation + "''" + "``"
# date_pattern = r'(\d{2})/(\d{2})/(\d{4})'


def preprocess(text_doc):
    # Contraction handling
    contractions_text = contractions.fix(text_doc)
    # Spell checking
    spell_text = spell_check(contractions_text)
    # Tokenization
    tokens_text = word_tokenize(spell_text)
    # Covert to lower or upper case
    lower_text = [token.lower() for token in tokens_text]
    # Removing Punctuations ... ex: [!,?,;,:, ...]
    punctuations_text = [token for token in lower_text if token.isalpha()]
    # Removing Stop Words ... ex: [a,an,the,over, ...]
    stop_words_text = [token for token in punctuations_text if token not in stop_words]
    # Stemming: Removing prefixes and suffixes ... ex: concentration -> concentr
    stemmed_text = [stemmer.stem(token) for token in stop_words_text]
    # POS tagging
    pos_tokens = pos_tag(stemmed_text)
    # Lemmatization: aims to obtain the base or dictionary form of a word ... ex: running -> run
    lemmatized_tokens = []
    for token, pos in pos_tokens:
        if pos.startswith('J'):  # Adjective
            lemmatized_token = lemmatizer.lemmatize(token, wordnet.ADJ)
        elif pos.startswith('V'):  # Verb
            lemmatized_token = lemmatizer.lemmatize(token, wordnet.VERB)
        elif pos.startswith('N'):  # Noun
            lemmatized_token = lemmatizer.lemmatize(token, wordnet.NOUN)
        elif pos.startswith('R'):  # Adverb
            lemmatized_token = lemmatizer.lemmatize(token, wordnet.ADV)
        else:
            lemmatized_token = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemmatized_token)
    # Removing auxiliary verbs
    aux_less_text = remove_auxiliary_verbs(lemmatized_tokens)
    # Remove single characters
    single_char_removal = [token for token in aux_less_text if len(token) > 1]
    normalized_text = ' '.join(single_char_removal)
    return normalized_text


def remove_auxiliary_verbs(tokens):
    # List of auxiliary verbs
    aux_verbs = ['am', 'is', 'are', 'was', 'were', 'being', 'been',
                 'have', 'has', 'had', 'do', 'does', 'did',
                 'can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must', 'ought']

    filtered_tokens = [token for token in tokens if token.lower() not in aux_verbs]

    return filtered_tokens


def spell_check(text):
    spell = Speller(lang='en')
    tokens = word_tokenize(text)

    # Perform POS tagging
    tagged_tokens = pos_tag(tokens)

    corrected_tokens = []
    for word, pos in tagged_tokens:
        # Skip spell-checking for nouns
        if pos.startswith('NN'):
            corrected_tokens.append(word)
        else:
            corrected_tokens.append(spell(word))

    corrected_text = ' '.join(corrected_tokens)
    return corrected_text


