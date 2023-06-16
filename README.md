### CareerPal

Welcome to CareerPal! We are Anand Advani, Shyla Bisht, Jeb Cui, and Diya Dinesh, and we present for the Databricks LLM Hackathon 2023 an application to help people in their careers and job applications. The website contains a tool to parse skills from a resume (which also helps applicants determine what automated resume parsing systems might find), and a chatbot trained on text from the Workplace Stackexchange to provide answers to career- and job-related questions. The chatbot also provides links to the Stackexchange posts that inform its response, which provides a measure of explainability and links to more information for the user.

In order to run the application and interact with the resume parser and chatbot, run these commands in a new directory in a shell:
1. `git clone https://github.com/SixtusTheSixth/career-pal.git`
2. `pip install -r requirements.txt`
3. `export FLASK_APP=flaskapp.py` (on POSIX) or `set FLASK_APP=flaskapp.py` (on Windows)
4. `flask run`
5. and open localhost:5000 (the link will appear in the terminal).

We hope you find this resource helpful!
