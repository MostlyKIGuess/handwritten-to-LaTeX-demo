# handwritten-to-LaTeX-demo

User can draw and brrrr

## Config Dataset

Windows:
```sh
cd .\Backend\learner\datasets\
kaggle datasets download xainano/handwrittenmathsymbols
tar -xvzf .\handwrittenmathsymbols.zip
tar -xvzf .\data.rar
```

Linux:
```sh
cd ./Backend/learner/datasets
kaggle datasets download xainano/handwrittenmathsymbols
unzip ./handwrittenmathsymbols.zip
rar x data.rar

- Very important make sure the extracted_images has 82 categories, and that will only happen after you extract data.rar .

## Training Instructions:

- After installing all the requiremnets by

```
pip install -r requiremnets.txt
```

make sure you are in the Backend directory.

- And run:

```
python trainmodel.py
```
