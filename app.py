import streamlit as st
from transformers import BartForSequenceClassification, BartTokenizer
import torch
import numpy as np
import contextlib
import plotly.express as px
import pandas as pd
from PIL import Image
import datetime

with open("hit_log.txt", mode='a') as file:
    file.write(str(datetime.datetime.now()) + '\n')

MODEL_DESC = {
    'Bart MNLI': """Bart with a classification head trained on MNLI.\n\nSequences are posed as NLI premises and topic labels are turned into premises, i.e. `business` -> `This text is about business.`""",
    'Bart MNLI + Yahoo Answers': """Bart with a classification head trained on MNLI and then further fine-tuned on Yahoo Answers topic classification.\n\nSequences are posed as NLI premises and topic labels are turned into premises, i.e. `business` -> `This text is about business.`""",
}

ZSL_DESC = """Recently, the NLP science community has begun to pay increasing attention to zero-shot and few-shot applications, such as in the [paper from OpenAI](https://arxiv.org/abs/2005.14165) introducing GPT-3. This demo shows how 🤗 Transformers can be used for zero-shot topic classification, the task of predicting a topic that the model has not been trained on."""

CODE_DESC = """```python
# pose sequence as a NLI premise and label as a hypothesis
from transformers import BartForSequenceClassification, BartTokenizer
nli_model = BartForSequenceClassification.from_pretrained('bart-large-mnli')
tokenizer = BartTokenizer.from_pretrained('bart-large-mnli')

premise = sequence
hypothesis = f'This text is about {label}.'

# run through model pre-trained on MNLI
x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                        max_length=tokenizer.max_len,
                        truncation_strategy='only_first')
logits = nli_model(x.to(device))[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(1)
prob_label_is_true = probs[:,1]
```"""

model_ids = {
    'Bart MNLI': 'bart-large-mnli',
    'Bart MNLI + Yahoo Answers': './bart_mnli_topics'
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@st.cache(allow_output_mutation=True)
def load_models():
    return {id: BartForSequenceClassification.from_pretrained(id).to(device) for id in model_ids.values()}

models = load_models()


@st.cache(allow_output_mutation=True)
def load_tokenizer(tok_id):
    return BartTokenizer.from_pretrained(tok_id)

@st.cache(allow_output_mutation=True, show_spinner=False)
def classify_candidate(nli_model_id, sequence, label, do_print_code):
    nli_model = models[nli_model_id]
    tokenizer = load_tokenizer('bart-large')

    # pose sequence as a NLI premise and label as a hypothesis
    premise = sequence
    hypothesis = f'This text is about {label}.'
    
    # run through model pre-trained on MNLI
    x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                            max_length=tokenizer.max_len,
                            truncation_strategy='only_first')
    with torch.no_grad():
        logits = nli_model(x.to(device))[0]

    # we throw away "neutral" (dim 1) and take the probability of
    # "entailment" (2) as the probability of the label being true 
    entail_contradiction_logits = logits[:,[0,2]]
    probs = entail_contradiction_logits.softmax(1)
    prob_label_is_true = probs[:,1]

    return prob_label_is_true.cpu()

def get_most_likely(nli_model_id,  sequence, labels, do_print_code):
    predictions = []
    for label in labels:
        predictions.append(classify_candidate(nli_model_id, sequence, label, do_print_code))
        do_print_code = False #only print code once per run
    predictions = torch.cat(predictions)
    
    most_likely = predictions.argsort().numpy()
    top_topics = np.array(labels)[most_likely]
    scores = predictions[most_likely].detach().numpy()
    return top_topics, scores

@st.cache(allow_output_mutation=True)
def get_sentence_model(model_id):
    return SentenceTransformer(model_id)

def load_examples():
    df = pd.read_json('texts.json')
    names = df.name.values.tolist()
    mapping = {df['name'].iloc[i]: (df['text'].iloc[i], df['labels'].iloc[i]) for i in range(len(names))}
    names.append('Custom')
    mapping['Custom'] = ('', '')
    return names, mapping

def plot_result(top_topics, scores):
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

    ex_names, ex_map = load_examples()

    logo = Image.open('huggingface_logo.png')
    st.sidebar.image(logo, width=120)
    st.sidebar.markdown(ZSL_DESC)
    model_desc = st.sidebar.selectbox('Model', list(MODEL_DESC.keys()), 0)
    do_print_code = st.sidebar.checkbox('Show code snippet', False)
    st.sidebar.markdown('#### Model Description')
    st.sidebar.markdown(MODEL_DESC[model_desc])
    st.sidebar.markdown('Originally proposed by [Yin et al. (2019)](https://arxiv.org/abs/1909.00161). Read more in our [blog post](https://joeddav.github.io/blog/2020/05/29/ZSL.html).')
    
    st.title('Zero Shot Topic Classification')
    example = st.selectbox('Choose an example', ex_names)
    height = min((len(ex_map[example][0].split()) + 1) * 2, 200)
    sequence = st.text_area('Text', ex_map[example][0], key='sequence', height=height)
    labels = st.text_input('Possible topics (comma-separated)', ex_map[example][1], max_chars=1000)

    labels = list(set([x.strip() for x in labels.strip().split(',') if len(x.strip()) > 0]))
    if len(labels) == 0 or len(sequence) == 0:
        st.write('Enter some text and at least one possible topic to see predictions.')
        return

    if do_print_code:
        st.markdown(CODE_DESC)

    model_id = model_ids[model_desc]

    with st.spinner('Classifying...'):
        top_topics, scores = get_most_likely(model_id, sequence, labels, do_print_code)

    plot_result(top_topics[-10:], scores[-10:])






if __name__ == '__main__':
    main()

