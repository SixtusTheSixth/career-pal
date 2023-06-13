from webapp import webapp
from flask import render_template

@webapp.route('/')
@webapp.route('/index')
def index():
	'''
	user = {'username': 'Aquila'}
	posts = [
		{
			'author': {'username' : 'John'},
			'body': 'Beatiful day in Portland!'
		},
		{
			'author': {'username': 'Susan'},
			'body': 'The Avengers movie was so cool!'
		}
	]
	'''
	return render_template('index.html', title='CareerPal')
