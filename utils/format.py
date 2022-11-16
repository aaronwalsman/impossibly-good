import os
import json
import numpy
import re
import torch
import torch_ac
import gym

import utils


def get_obss_preprocessor(obs_space, image_dtype=torch.float):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(
                    obss, device=device, dtype=image_dtype)
            })

    # Check if it is a MiniGrid observation space
    elif (
        isinstance(obs_space, gym.spaces.Dict) and
        "image" in obs_space.spaces.keys()
    ):
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])

        def preprocess_obss(obss, device=None):
            processed = {
                "image": preprocess_images(
                    [obs["image"] for obs in obss],
                    device=device,
                    dtype=image_dtype
                ),
            }
                #"text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            if "observed_color" in obss[0]:
                processed["observed_color"] = preprocess_long(
                    [obs['observed_color'] for obs in obss], device=device)
            if "expert" in obss[0]:
                processed["expert"] = preprocess_long(
                    [obs['expert'] for obs in obss], device=device)
            if "step" in obss[0]:
                processed["step"] = preprocess_long(
                    [obs['step'] for obs in obss], device=device)
            if "switching_time" in obss[0]:
                processed["switching_time"] = preprocess_long(
                    [obs['switching_time'] for obs in obss], device=device)
            return torch_ac.DictList(processed)
        preprocess_obss.vocab = vocab
    
    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None, dtype=torch.float):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=dtype)

def preprocess_long(x, device=None):
    return torch.tensor(x, device=device, dtype=torch.long)

def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]
