import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def add_text(history, text):
    history = history + [[text, None]]
    return history, gr.update(value="", interactive=False)

def process_input(history):
    inp = history[-1][0]
    # response = "I have received your input, which is: \n" + inp
    if len(history) <= 2:
       chat_bot.reset_context()
    response = chat_bot.chat(inp)
    history[-1][1] = response
    return history

def build_qa_chain():
  torch.cuda.empty_cache()
  # Defining our prompt content.
  # langchain will load our similar documents as {context}
  template = """You are a chatbot having a conversation with a human. You are asked to answer career questions, and you are helping the human apply for jobs.
  Given the following extracted parts of a long document and a question, answer the user question. If you don't know, say that you do not know. 
  
  {context}
 
  {chat_history}
 
  {human_input}
 
  Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'human_input', 'chat_history'], template=template)
 
  # Increase max_new_tokens for a longer response
  # Other settings might give better results! Play around
  model_name = "databricks/dolly-v2-3b" # can use dolly-v2-3b, dolly-v2-7b or dolly-v2-12b for smaller model and faster inferences.
  instruct_pipeline = pipeline(model=model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", 
                               return_full_text=True, max_new_tokens=256, top_p=0.95, top_k=50)
  hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)
 
  # Add a summarizer to our memory conversation
  # Let's make sure we don't summarize the discussion too much to avoid losing to much of the content
 
  # Models we'll use to summarize our chat history
  # We could use one of these models: https://huggingface.co/models?filter=summarization. facebook/bart-large-cnn gives great results, we'll use t5-small for memory
  summarize_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
  summarize_tokenizer = AutoTokenizer.from_pretrained("t5-small", padding_side="left", model_max_length = 512)
  pipe_summary = pipeline("summarization", model=summarize_model, tokenizer=summarize_tokenizer) #, max_new_tokens=500, min_new_tokens=300
  # langchain pipeline doesn't support summarization yet, we added it as temp fix in the companion notebook _resources/00-init 
  hf_summary = HuggingFacePipeline(pipeline=pipe_summary)
  #will keep 500 token and then ask for a summary. Removes prefix as our model isn't trained on specific chat prefix and can get confused.
  memory = ConversationSummaryBufferMemory(llm=hf_summary, memory_key="chat_history", input_key="human_input", max_token_limit=500, human_prefix = "", ai_prefix = "")
 
  # Set verbose=True to see the full prompt:
  print("loading chain, this can take some time...")
  return load_qa_chain(llm=hf_pipe, chain_type="stuff", verbose=True, prompt=prompt, memory=memory)

class ChatBot():
  def __init__(self, db):
    self.reset_context()
    self.db = db
 
  def reset_context(self):
    self.sources = []
    self.discussion = []
    # Building the chain will load Dolly and can take some time depending on the model size and your GPU
    self.qa_chain = build_qa_chain()
 
  def get_similar_docs(self, question, similar_doc_count):
    return self.db.similarity_search(question, k=similar_doc_count)
 
  def chat(self, question):
    # Keep the last 3 discussion to search similar content
    self.discussion.append(question)
    similar_docs = self.get_similar_docs(" \n".join(self.discussion[-3:]), similar_doc_count=2)
    # Remove similar doc if they're already in the last questions (as it's already in the history)
    similar_docs = [doc for doc in similar_docs if doc.metadata['source'] not in self.sources[-3:]]
 
    result = self.qa_chain({"input_documents": similar_docs, "human_input": question})
    # Cleanup the answer for better display:
    answer = result['output_text'].strip().capitalize()
    result_html = f"<p><blockquote style=\"font-size:18px\">{answer}</blockquote></p>"
    result_html += "<p><hr/></p>"
    for d in result["input_documents"]:
      source_id = d.metadata["source"]
      self.sources.append(source_id)
      result_html += f"<p>(Source: <a href=\"https://workplace.stackexchange.com/a/{source_id}\">{source_id}</a>)</p>"
    return result_html

with gr.Blocks() as demo:
    global chat_bot
    workplace_vector_db_path = "workplace_db"

    hf_embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    chroma_db = Chroma(collection_name="workplace_docs", embedding_function=hf_embed, persist_directory=workplace_vector_db_path)
    chat_bot = ChatBot(chroma_db)
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

demo.launch() # server_port=7860, show_api=False, share=False, inline=True) # , share = True, inline = True)
