## Using Machine Learning to explore decarbonisation strategies
This project aims to utilise machine learning techniques such as pdf text extraction, image processing and text classification to generate insights related to the environmental aspect of ESG. Through this process, we will be extracting textual and table information from sustainability reports

## Getting Started
This project utilises a combination of python scripts and open sourced packages, tested on Python 3.9.x to run smoothly. For the best experience, we suggestt to run this project on a Windows-based PC.

To start this project, you can either clone this repository to your local computer, or download this repository as a zip file. Upon installation, please run the following commands in either the command line or the terminal:

`pip install /pth/requirements.txt`

where pth is the path to reach requirements.txt.

In addition, we recommend installing the [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2/tree/main) and store them in the `pipeline/bert_model` folder to reduce initial set-up time. Please specify `Y` for yes and `N` for no when running the application so that the application can run smoothly.

To run the application, please use either of the following command with either the command prompt or the terminal:

`python main.py`

`python3 main.py`

and follow the instructions displayed on the command prompt/terminal.

## Dashboard
The outputs of this python process will be put into our dashboard to generate better insights on both company-based analysis and sector-based analysis. Our dashboard and the installation instrutions are linked in this Github below:

[Dashboard](https://github.com/fancasta/Capstone-Dashboard)

## Built with
* [gensim](https://github.com/RaRe-Technologies/gensim)
* [keras](https://keras.io/)
* [nltk](https://www.nltk.org/)
* [pdf2image](https://github.com/Belval/pdf2image)
* [pdfminer.six](https://pdfminersix.readthedocs.io/en/latest/)
* [pdfminer3](https://github.com/gwk/pdfminer3)
* [PyPDF2](https://pypdf2.readthedocs.io/en/latest/)
* [pymupdf](https://pymupdf.readthedocs.io/en/latest/toc.html)
* [pytorch](https://pytorch.org/)
* [scikit_learn](https://scikit-learn.org/)
* [scipy](https://scipy.org/)
* [spacy](https://spacy.io/)
* [tabula-py](https://tabula-py.readthedocs.io/en/latest/)
* [tensorflow](https://www.tensorflow.org/)
* [transformers](https://pytorch.org/hub/huggingface_pytorch-transformers/)

We would also like to thank Team08 for their help in the pdf text extraction process.

* [Github](https://github.com/jaokuean/team08-capstone)

## Contributors
* Png Chen Wei
* Ng Shu Hui
* Nguyen An Khanh
* Mu Yuchen
* Sie Jie Xiang
