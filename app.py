import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import numpy as np
import contextlib
import plotly.express as px
import pandas as pd
from PIL import Image
import datetime
import os
import psutil

with open("hit_log.txt", mode='a') as file:
    file.write(str(datetime.datetime.now()) + '\n')

MODEL_DESC = {
    'Bart MNLI': """Bart with a classification head trained on MNLI.\n\nSequences are posed as NLI premises and topic labels are turned into premises, i.e. `business` -> `This text is about business.`""",
    'Bart MNLI + Yahoo Answers': """Bart with a classification head trained on MNLI and then further fine-tuned on Yahoo Answers topic classification.\n\nSequences are posed as NLI premises and topic labels are turned into premises, i.e. `business` -> `This text is about business.`""",
    'XLM Roberta XNLI (cross-lingual)': """XLM Roberta, a cross-lingual model, with a classification head trained on XNLI. Supported languages include: _English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili, and Urdu_.

Note that this model seems to be less reliable than the English-only models when classifying longer sequences.

Examples were automatically translated and may contain grammatical mistakes.

Sequences are posed as NLI premises and topic labels are turned into premises, i.e. `business` -> `This text is about business.`""",
}

ZSL_DESC = """Recently, the NLP science community has begun to pay increasing attention to zero-shot and few-shot applications, such as in the [paper from OpenAI](https://arxiv.org/abs/2005.14165) introducing GPT-3. This demo shows how 🤗 Transformers can be used for zero-shot topic classification, the task of predicting a topic that the model has not been trained on."""

CODE_DESC = """```python
from transformers import pipeline
classifier = pipeline('zero-shot-classification',
                      model='{}')
hypothesis_template = 'This text is about {{}}.' # the template used in this demo

classifier(sequence, labels,
           hypothesis_template=hypothesis_template,
           multi_class=multi_class)
# {{'sequence' ..., 'labels': ..., 'scores': ...}}
```"""

model_ids = {
    'Bart MNLI': 'facebook/bart-large-mnli',
    'Bart MNLI + Yahoo Answers': 'joeddav/bart-large-mnli-yahoo-answers',
    'XLM Roberta XNLI (cross-lingual)': 'joeddav/xlm-roberta-large-xnli'
}

device = 0 if torch.cuda.is_available() else -1

@st.cache(allow_output_mutation=True)
def load_models():
    return {id: AutoModelForSequenceClassification.from_pretrained(id) for id in model_ids.values()}

models = load_models()


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_tokenizer(tok_id):
    return AutoTokenizer.from_pretrained(tok_id)

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_most_likely(nli_model_id, sequence, labels, hypothesis_template, multi_class, do_print_code):
    classifier = pipeline('zero-shot-classification', model=models[nli_model_id], tokenizer=load_tokenizer(nli_model_id), device=device)
    outputs = classifier(sequence, labels, hypothesis_template, multi_class)
    return outputs['labels'], outputs['scores']

def load_examples(model_id):
    model_id_stripped = model_id.split('/')[-1]
    df = pd.read_json(f'texts-{model_id_stripped}.json')
    names = df.name.values.tolist()
    mapping = {df['name'].iloc[i]: (df['text'].iloc[i], df['labels'].iloc[i]) for i in range(len(names))}
    names.append('Custom')
    mapping['Custom'] = ('', '')
    return names, mapping

def plot_result(top_topics, scores):
    top_topics = np.array(top_topics)
    scores = np.array(scores)
    scores *= 100
    fig = px.bar(x=scores, y=top_topics, orientation='h', 
                 labels={'x': 'Confidence', 'y': 'Label'},
                 text=scores,
                 range_x=(0,115),
                 title='Top Predictions',
                 color=np.linspace(0,1,len(scores)),
                 color_continuous_scale='GnBu')
    fig.update(layout_coloraxis_showscale=False)
    fig.update_traces(texttemplate='%{text:0.1f}%', textposition='outside')
    st.plotly_chart(fig)

        

def main():
    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    logo = Image.open('huggingface_logo.png')
    st.sidebar.image(logo, width=120)
    st.sidebar.markdown(ZSL_DESC)
    model_desc = st.sidebar.selectbox('Model', list(MODEL_DESC.keys()), 0)
    do_print_code = st.sidebar.checkbox('Show code snippet', False)
    st.sidebar.markdown('#### Model Description')
    st.sidebar.markdown(MODEL_DESC[model_desc])
    st.sidebar.markdown('Originally proposed by [Yin et al. (2019)](https://arxiv.org/abs/1909.00161). Read more in our [blog post](https://joeddav.github.io/blog/2020/05/29/ZSL.html).')
    
    model_id = model_ids[model_desc]
    ex_names, ex_map = load_examples(model_id)

    st.title('Zero Shot Topic Classification')
    example = st.selectbox('Choose an example', ex_names)
    height = min((len(ex_map[example][0].split()) + 1) * 2, 200)
    sequence = st.text_area('Text', ex_map[example][0], key='sequence', height=height)
    labels = st.text_input('Possible topics (separated by `,`)', ex_map[example][1], max_chars=1000)
    multi_class = st.checkbox('Allow multiple correct topics', value=True)

    hypothesis_template = "This text is about {}."

    labels = list(set([x.strip() for x in labels.strip().split(',') if len(x.strip()) > 0]))
    if len(labels) == 0 or len(sequence) == 0:
        st.write('Enter some text and at least one possible topic to see predictions.')
        return

    if do_print_code:
        st.markdown(CODE_DESC.format(model_id))

    with st.spinner('Classifying...'):
        top_topics, scores = get_most_likely(model_id, sequence, labels, hypothesis_template, multi_class, do_print_code)

    plot_result(top_topics[::-1][-10:], scores[::-1][-10:])

    if "socat" not in [p.name() for p in psutil.process_iter()]:
        os.system('socat tcp-listen:8000,reuseaddr,fork tcp:localhost:8001 &')






if __name__ == '__main__':
    main()

