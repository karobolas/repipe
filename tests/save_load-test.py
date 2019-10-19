import repipe.pipeline as pipeline
from repipe.serializeable import Serializable


pipe = pipeline.Pipeline(
    steps=[
        pipeline.TransformStep(
            in_fields='text',
            out_field='text_scrubbed',
            transform=pipeline.TextScrubber(
                lower=True,
                tokenize=False,
                filters='()[]{}<>$&%#|-+=*_─…•—–"\'’/\\“ °´®”̈~¿'
            )
        ),
        pipeline.TransformStep(
            in_fields='text_scrubbed',
            out_field='tokenized',
            transform=pipeline.KerasTokenizerAdapter(
                filters=''
            )
        ),
        pipeline.TransformStep(
            in_fields='tokenized',
            out_field='padded_tokenized',
            transform=pipeline.KerasPadSequencesAdapter(
                maxlen=750,
                padding='post',
                truncating='post',
                value=0,
                dtype='i4'
            )
        ),
        pipeline.FeatureSelector(
            features=[
                'padded_tokenized'
            ]
        )
    ]
)

import yaml

# Save pipe
with open(f'my_pipe.yaml', 'w') as f:
    yaml.safe_dump(pipe.to_dict(), f)

# Load pipe
with open(f'my_pipe.yaml', 'r') as f:
    pipe2 = Serializable.load(yaml.safe_load(f))


print(pipe.to_dict())
print()
print(pipe2.to_dict())
assert pipe.to_dict() == pipe2.to_dict()