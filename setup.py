from setuptools import setup, find_packages

setup(
    name='re-pipe',
    version='0.0.1',
    description='A reproducible data/NLP pipeline',
    url='https://github.com/stdexcept/repipe',
    author='Ali Mosavian',
    author_email='ali@octai.se',

    python_requires='>=3.6',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        'gensim == 3.8.3',
        'joblib == 0.14.0',
        'Keras-Preprocessing == 1.1.0',
        'nltk == 3.4.5',
        'numpy == 1.17.3',
        'pandas == 0.25.2',
        'scikit-learn == 0.23.2',
        'scipy == 1.4.1',
    ],
    extras_require={
        'test': [
            'pyyaml==5.1.2',
            'nose==1.3.7'
        ],
    }
)
