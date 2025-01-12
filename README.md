This is a leetcode solver project all from gathering the data to fine tuning it with GPT2.

# Gathering some data

Used 2 kaggle datasets containing questions title along with their text and solution link. Combined them both in order to get non-overlapping problems and gather a larger dataset.
https://www.kaggle.com/datasets/manthansolanki/leetcode-questions
https://www.kaggle.com/datasets/manthansolanki/leetcode-questions

As for the solutions, I took advantage of a github repo containing most if not all solutions written in python3, this facilitates the jobs since I don't have to waste time and computational power for scraping it myself.
https://github.com/cnkyrpsgl/leetcode/blob/master/README.md

After getting my inputs and outputs, I've cleaned up the data, kept the columns I needed and mapped the problems to code solutions for easy integration.
Code in analyzing_data.ipynb
Just like that, we have around 1k problems, 600 more than my previous attempt when using a repo with around 400 code solutions.

# Fine tuning

A pretty straight forward 100 liner code was written in fine-tuner.py where we use Hugging Face's transformers library for the pre-trained model and PyTorch's helper functions for creating the dataloader, calculating the loss and optimizing the model.
After a couple of training tries, the loss per epoch would decrease by around 0.2. After 10 epochs, the model was horrible but will try soon with GPUs for faster and more training. If this doesn't do it, might have to tweak the hyperparameters to better off the training.

Hopefully it doesn't suck later on.
