

# flag-text-embedding-chinese 

Map any text to a 1024-dimensional dense vector space and can be used for tasks like retrieval, classification,  clustering, or semantic search.



## Usage (Sentence-Transformers)

Using this model becomes easy when you have [sentence-transformers](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer
sentences = ["样例数据-1", "样例数据-2"]

model = SentenceTransformer('Shitao/flag-text-embedding-chinese')
embeddings = model.encode(sentences, normalize_embeddings=True)
print(embeddings)
```



## Usage (HuggingFace Transformers)
Without [sentence-transformers](https://www.SBERT.net), you can use the model like this: First, you pass your input through the transformer model, then you have to apply the right pooling-operation on-top of the contextualized word embeddings.

```python
from transformers import AutoTokenizer, AutoModel
import torch


# Sentences we want sentence embeddings for
sentences = ["样例数据-1", "样例数据-2"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('Shitao/flag-text-embedding-chinese')
model = AutoModel.from_pretrained('Shitao/flag-text-embedding-chinese')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]

print("Sentence embeddings:")
print(sentence_embeddings)
```



## Evaluation Results

For an automated evaluation of this model, see the *Chinese Embedding Benchmark*: [link]()




## Citing & Authors

<!--- Describe where people can find more information -->