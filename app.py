import gradio as gr
from webapp import webapp

def add_text(history, text):
    history = history + [[text, None]]
    return history, gr.update(value="", interactive=False)

def process_input(history):
    inp = history[-1][0]
    response = "I have received your input, which is: \n" + inp
    history[-1][1] = response
    return history

with gr.Blocks() as demo:
    gr.Markdown('''
    ## **CareerPal**
    here to ease your anxiety about your future
    ''')
    with gr.Row():
        output_box = gr.Chatbot([[None, "Welcome! What can I help you with today?"]], show_label=False).style(height=450)
    with gr.Row(): # TODO: Box or Group instead of row?
        with gr.Column(scale=7):
            input_box = gr.Textbox(show_label=False, placeholder="Ask something here and press enter...").style(container=False)
        with gr.Column(scale=1):
            clear_btn = gr.Button(value="Clear")
    
    txt_msg = input_box.submit(add_text, inputs=[output_box, input_box], outputs=[output_box, input_box],
                               queue=False).then(process_input, output_box, output_box)
    txt_msg.then(lambda: gr.update(interactive=True), inputs=None, outputs=input_box, queue=False)

    clear_btn.click(lambda: None, inputs=None, outputs=output_box, queue=False)

demo.launch(server_port=7860, show_api=False, share=False) # , share = True, inline = True)

# set FLASK_APP=app.py
# flask run -h localhost -p 7860