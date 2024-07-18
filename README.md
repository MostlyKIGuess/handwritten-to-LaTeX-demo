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
```

- Very important make sure the extracted_images has 82 categories, and that will only happen after you extract data.rar .

## Training Instructions:

- After installing all the requiremnets by

```
pip install -r requirements.txt
```

make sure you are in the Backend directory.

- And run:

```
python trainmodel.py
```

## Electron build guide

User can build their own electron application to run frontend and backend together

### Install deps

```sh
npm install
```

### Run electron app

Before running the electron app, you need to build the frontend react app. To do that, follow these:

```sh
cd Frontend
npm run build
```

Now electron is ready to run:

```sh
cd .. # don't do that if you are not in frontend's parent directory
npm start
# or
npx electron .
```

### Build binary for your platform

As of now, build binary is facing some issue in starting, but might work on diff platforms
```sh
# for linux
npx electron-builder --linux deb tar.xz

# build for Windows ia32
npx electron-builder --win --ia32

# build for macOS, Windows and Linux
npx electron-builder -mwl
```