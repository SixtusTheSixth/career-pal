from webapp import webapp
from flask import render_template

@webapp.route('/')
@webapp.route('/index')
def index():
	return render_template('index.html', title='CareerPal')
