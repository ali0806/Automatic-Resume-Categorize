# Resume categorizing Based On Their Domain


This project focuses on categorizing resumes based on their text data using a BERT model. The project utilizes a hybrid architecture combining BERT with a custom classifier to perform the classification. The model is implemented in PyTorch and leverages various techniques to ensure effective training and evaluation.


## Table of Contents

- [POS and NER Tagging]()
  - [Table of Contents](#table-of-contents)
  - [model](#model)
  - [Datasets](#datasets)
  - [Setup](#setup)
  - [Training & Evaluation](#training--evaluation)
  - [Usage](#usage)
  - [License](#license)

## Model

This project uses a **BERT** architecture for categorizing resumes according their domain. It transforms BERT's output using a linear layer, applies dropout for regularization, and produces logits through a final linear layer to classify resumes into different categories.


- [**BERT(Pretrained)**](https://drive.google.com/file/d/1STMeS3o7mnD0yi4dw_W0CrStGw_CZ3cJ/view?usp=sharing)

## Datasets

We use a Resume dataset for training the model..
- [**Dataset**](https://drive.google.com/file/d/1S_QL3ELp1scyBIxGg52iuxBjeO1UAyRV/view?usp=sharing)

 
## Setup

For installing the necessary requirements, use the following bash snippet
```bash
$ git clone https://github.com/ali0806/Automatic-Resume-Categorize.git
$ cd Automatic-Resume-Categorize/
$ python -m venv env
$ source env/bin/activate # For Linux
$ pip install -r requirements.txt
```
* Use the newly created environment for running the scripts in this repository.

## Training & Evaluation

To see list of all available options, do  ```python train.py -h```

For Training model on single GPU, a minimal example is as follows:

```bash
$ python main.py \
 --batch_size 64 \
 --max_length 256 \
 --epochs 30 \
 --learning_rate 1e-5

```
**Note:** For navigate ```main.py``` first  go to ```src ``` folder with ```cd src``` command
* After training with default parameters

|     Tag          |   Accuracy   |     Precision     |      Recall     | F1-Score     |
|----------------|-----------|-----------|-----------|-----------|
| Model| 0.849|0.831|0.824|0.837|


## Usage
To begin, first download the checkpoint file 
from the provided link.
  - [Pretrain Model](https://drive.google.com/file/d/1STMeS3o7mnD0yi4dw_W0CrStGw_CZ3cJ/view?usp=sharing)

 After the download is complete, you need to move the checkpoint file to the `model` directory of project.
```bash
python script.py --file_path FILE_PATH 

```
It outputs the resumes by organizing them into separate directories based on their domains and generates a CSV file containing information about each file's name and its corresponding category. All this elements store on `output` folder
 
## License
Distributed under the MIT License. See LICENSE for more information. 





