# An LSTM Odyssey

Code for training variants of "LSTM: A Search Space Odyssey" on Fomoro.

Check out the [blog post](https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93).

## Training

### Cloud Setup

1. Follow the [installation guide](https://fomoro.gitbooks.io/guide/content/installation.html) for Fomoro.
2. Clone the repo: `git clone https://github.com/fomorians/lstm-odyssey.git && cd lstm-odyssey`
3. Create a new model: `fomoro model create`
4. Start training: `fomoro session start`
5. Follow the logs: `fomoro session logs -f`

### Local Setup

1. [Install TensorFlow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation).
2. Clone the repo: `git clone https://github.com/fomorians/lstm-odyssey.git && cd lstm-odyssey`
3. Run training: `python main.py`
