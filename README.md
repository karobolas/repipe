# What's this?
Re-pipe is a reproducible data pipeline (for now targeted at NLP data pipes). It's main benefit is that the 
whole pipe can be serialized and exactly reproduced, thus making it much more convenient to perform various 
experiments when developing an NLP model. 

# Why 
A common problem when performing many experiments or doing hyper-parameter search for a model is that it's hard to
keep track of what changes led to a specific model and to be able to reproduce it. Re-pipe enables the data 
transformation pipe to be serialized along with the end model and directly loaded into your production environment 
as one unit without making any mistakes.

# State
Re-pipe is not yet in a clean state and is under development. 

# Examples
### Creating a pipe

```python
import repipe.pipeline as pipeline

pipe = pipeline.Pipeline(
    steps=[
        pipeline.TransformStep(
            in_fields=['short_description','description'],
            out_field='text',
            transform=pipeline.TextFieldUnion()
        ),
        pipeline.TransformStep(
            in_fields='text',
            out_field='text_scrubbed',
            transform=pipeline.TextScrubber(
                lower=True,
                tokenize=False,
                filters='()[]{}<>$&%#|-+=*_"\'’/\\®~¿'
            )
        ),
        pipeline.TransformStep(
            in_fields='company',
            out_field='company_onehot',
            transform=pipeline.OneHotEncoderAdapter(
                sparse=True,
                categories=dataset.company.fillna('').unique(),
            )
        ),
        pipeline.TransformStep(
            in_fields='contact_type',
            out_field='contact_type_onehot',
            transform=pipeline.OneHotEncoderAdapter(
                sparse=True,
                categories=dataset.contact_type.fillna('').unique()                
            )
        ),
        pipeline.TransformStep(
            in_fields='text_scrubbed',
            out_field='word_hashes',
            transform=pipeline.HashingVectorizerAdapter(
                analyzer='word',
                lowercase=True,
                n_features=7500,
                ngram_range=(1,1)            
            )
        ),
        pipeline.TransformStep(
            in_fields='text_scrubbed',
            out_field='char_3grams',
            transform=pipeline.HashingVectorizerAdapter(
                analyzer='char_wb',
                lowercase=True,
                n_features=7500,
                ngram_range=(3,3)
            )
        ),
        pipeline.TransformStep(
            in_fields='text_scrubbed',
            out_field='embeddings',
            transform=pipeline.WordVectorEmbedder(
                path='models/word2vec.model.2019-07-02.bin',
                max_embedding_len=64*2,
                dtype='float16'
            )
        ),
        pipeline.FeatureSelector(
            features=[
                'company_onehot',
                'contact_type_onehot',                
                'word_hashes',
                'char_3grams',
                'embeddings'
            ]
        )
    ]
)
```

### Using the pipe

```python
x_trainT = pipe.transform(x_train)
x_testT = pipe.transform(x_test)
```

### Saving/loading the pipe
```python
import yaml
from repipe.serializable import Serializable
 
# Save pipe
with open(f'my_pipe.yaml', 'w') as f:
    yaml.dump(pipe.to_dict(), f)

# Load pipe
with open(f'my_pipe.yaml', 'r') as f:
    pipe2 = Serializable.load(yaml.load(f))
    
# Use
X = pipe2.transform(dataset)
```