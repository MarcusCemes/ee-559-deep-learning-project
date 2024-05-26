# A Robotic System Powered by Deep Learning

_EE-559 Deep Learning @ EPFL_

Authors: Marcus Cemes, Osman Kiziltug, Lluka Stojollari

Hate speech detection is a vast and increasingly relevant topic in the digital age, where the easy access and anonymity of online communication could be playing a role in transforming the way we interact and express ourselves \cite{online-offline-hate}. A limitation in the current landscape is that many models are trained on datasets sourced from online social media platforms where this type of detection is already employed and are often subject to abbreviated writing styles and internet slang.

In this project, we asked ourselves the following question: **how well would the currently employed techniques and available datasets translate into a more physical experience**? More specifically, we wanted to investigate the feasibility of a robotics learning system that could engage in a limited form of free "conversation" with a user, exhibiting a sentiment-based actuation based on the user’s input, such as moving away or shaking its head in dismayed response to what is perceived as detected hate speech.

Our objective was to take state-of-the-art audio transcription and natural language models and integrate these with the Thymio educational robot. We hypothesised that such a system could provide a more tangible and real-time response to the end-user, thereby enhancing their awareness of the impact of their words through a more sociably understandable response. In the following sections, we delve into the specifics of our methodology, the challenges we faced and the results of our experiment.

_For more information, see the associated report._

## Prerequesites

Install a recent version of Python (3.8 or later) and the required packages. We recommend using a virtual environment to avoid conflicts with other projects.

_Windows_

```sh
python -m venv .venv
.venv/Scripts/activate.bat
pip install -r requirements.txt
```

If Pytorch is not correctly installed, use their website instructions:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

If you would like to connect to a Thymio robot, ensure that the driver is running (Thymio suite is open).

You will need to clone the hateBERT repository:

```sh
mkdir tmp
cd tmp
git clone https://huggingface.co/GroNLP/hateBERT
```

Additionally, you will need the binary and multi-class classifier weights. These are not included with this repository due to their size. Place them in the `tmp` folder, and ensure the paths match the configuration in `app/analysis.py`.

### Web UI

If you would like to connect to a web UI, you will need to install Node.js and build the frontend.

```sh
cd ui
pnpm install --frozen-lockfile
pnpm build
```

## Running

To run the application, execute the following command:

```sh
python -m app
```

This will connect to the Thymio robot, start a web server and run the main application and enter an interactive prompt. To execute a single run, type `prompt` and press enter. This will start recording an audio sample from the microphone, transcribe it, classify it and actuate the robot accordingly. To see all available commands, type `help`.

If you built the web UI, you can access it on the local server at `http://localhost:8080`.

All source code is contained within the `app` folder. The application can be configured by toggling the constants at the top of each file.

## Training code

The code used to train the binary and multi-class classifiers is available in the Jupyter notebooks

The training process uses the [Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) dataset from UC Berkeley, located in the `data` folder.

Additional training code, designed to run on the EPFL SCITAS GPU compute cluster using SLURM is located in the `training` folder. This was deprecated in favour of using two seperated classifier models.
