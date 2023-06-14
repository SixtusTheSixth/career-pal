from webapp import webapp
from flask import render_template, request
import docx2txt
import nltk
import requests
import time

nltk.download('stopwords')
nltk.download('punkt')

#education
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

RESERVED_WORDS = [
    'school',
    'high school',
    'college',
    'univers',
    'university',
    'academy',
    'faculty',
    'institute',
    'polytechnic',
    'certificate',
    'undergraduate',
    'graduate',
    'phd'
]
 

def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        return txt.replace('\t', ' ')
    return None

def skill_exists(skill):
    url = f'https://api.apilayer.com/skills?q={skill}&amp;count=1'
    headers = {'apikey': 'l2BFkm2hY9qeQT37xaCB1rdyeWUnY5S1'}
    response = requests.request('GET', url, headers=headers)
    result = response.json()
 
    if response.status_code == 200:
        return len(result) > 0 and result[0].lower() == skill.lower()
    raise Exception(result.get('message'))

def extract_skills(input_text, timerange):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(input_text)
 
    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]
 
    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]
 
    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))
 
    # we create a set to keep the results in.
    found_skills = set()
    #print(found_skills)
	
    starttime = time.time()
	# we search for each token in our skills database
    for token in filtered_tokens:
        if skill_exists(token.lower()):
            print(token)
            found_skills.add(token.lower())
        if time.time() - starttime > timerange:
            return found_skills
 
    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if skill_exists(ngram.lower()):
            print(ngram)
            found_skills.add(ngram.lower())
        if time.time() - starttime > timerange:
            return found_skills
 
    return found_skills

def extract_education(input_text):
    organizations = []
 
    # first get all the organization names using nltk
    for sent in nltk.sent_tokenize(input_text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk, 'label') and chunk.label() == 'ORGANIZATION':
                organizations.append(' '.join(c[0] for c in chunk.leaves()))
    print(organizations)
    # we search for each bigram and trigram for reserved words
    # (college, university etc...)
    education = set()
    for org in organizations:
        for word in RESERVED_WORDS:
            if org.lower().find(word) >= 0:
                education.add(org)
 
    return education

@webapp.route('/')
@webapp.route('/index')
def index():
	return render_template('index.html', title='CareerPal')

@webapp.route('/action_page', methods=['POST'])
def action_page():
	f = request.files['filename']
	text = extract_text_from_docx(f)
	skills = extract_skills(text, 40)
	education = extract_education(text)
	return render_template('index.html', title='CareerPal', skills=skills, education=education)