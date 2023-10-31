from datasets import load_dataset
import torch
from transformers import AutoTokenizer
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric
import numpy as np
from sklearn.metrics import confusion_matrix
from transformers import Trainer
from transformers import pipeline

def bert(sentence):

    base_url = './'


    dataset = load_dataset('csv', data_files={'train': base_url+'bert_train.csv','validation': base_url+'bert_val.csv','test': base_url+'bert_test.csv'})

    # import torch

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()



    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    # from transformers import AutoTokenizer

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)   

    # import pandas as pd

    emoji_df = pd.read_csv('emoji.csv')
    emoji_list = emoji_df['emoji'].to_list()
    print(emoji_list)

    vocab_df = pd.read_csv('vocab.csv')
    vocab_list = vocab_df['word'].to_list()
    print(vocab_list)

    new_tokens = emoji_list + vocab_list

    num_added_toks = tokenizer.add_tokens(new_tokens)
    print('We have added', num_added_toks, 'tokens')
    # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.

    def tokenize_function(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    dataset_encoded = dataset.map(tokenize_function, batched=True, batch_size=None)
    dataset_encoded
    print(tokenizer.tokenize('finally my first politician blocked me i am feeling of myself right now 😆'))

    print(dataset_encoded['train'][17])

    # from transformers import AutoModelForSequenceClassification

    num_labels = 3
    model = (AutoModelForSequenceClassification
            .from_pretrained(checkpoint, num_labels=num_labels)
            .to(device))
    model.resize_token_embeddings(len(tokenizer))

    # from transformers import Trainer, TrainingArguments

    batch_size = 16
    logging_steps = len(dataset_encoded["train"]) // batch_size
    model_name = f"{checkpoint}-finetuned"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=5,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    log_level="error",
                                    optim='adamw_torch'
                                    )

    # from datasets import load_metric
    # import numpy as np
    metric = load_metric('accuracy')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    # from transformers import Trainer

    torch.cuda.empty_cache()

    trainer = Trainer(model=model,
                    compute_metrics=compute_metrics,
                    args=training_args, 
                    train_dataset=dataset_encoded["train"],
                    eval_dataset=dataset_encoded["validation"],
                    tokenizer=tokenizer)
    trainer.train() 

    trainer.evaluate(dataset_encoded["train"])

    trainer.evaluate(dataset_encoded["test"])

    output=trainer.predict(dataset_encoded["test"])[0]
    predictions = np.argmax(output, axis=1)

    # from sklearn.metrics import confusion_matrix

    cm=confusion_matrix(dataset_encoded["test"]["label"],predictions)
    trainer.save_model()

    # from transformers import pipeline
    checkpoint = "bert-base-uncased"
    model_name = f"{checkpoint}-finetuned"
    classifier = pipeline('text-classification', model=model_name)
    # sentence = "monkey pox is a hoax all the community hah 🤡 🏳️‍🌈"
    classifier_dict = {
        'LABEL_0' : 'NEUTRAL',
        'LABEL_1' : 'NEGATIVE',
        'LABEL_2' : 'POSITIVE'
    }

    c = classifier(sentence)
    a = c.split(" ")
    # print(f'\nSentence: {sentence}')
    print(f"\nThis sentence is classified with a {classifier_dict[c[0]['label']]} sentiment")
    return [classifier_dict[c[0]['label']],a]


from flask import Flask, send_file, make_response
from flask import Flask, request, render_template
app = Flask(__name__)


# dataset = load_dataset('csv', data_files={'train': base_url+'bert_train.csv','validation': base_url+'bert_val.csv','test': base_url+'bert_test.csv'})


@app.route('/test', methods = ['POST'])
def main():
  
   inp = request.get_json()['input']
   print(inp)
   c = bert(inp) 
   return {"result":c[0],"tokenize":c[1]}

  
if __name__ == '__main__':
    app.run(host="localhost",port=5005)
