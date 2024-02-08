# Better Uncensored Models, and How Dolphins are Trained Bigots

![Better Uncensored, the original censored list is quite biassed.](https://github.com/sudoaza/better-uncensored/blob/main/images/better-uncensored.png?raw=true)

"Uncensored" datasets and models based on them (like *-dolphin) have been haphazardly (or maliciously)
curated to remove examples of model refusals, and what the authors call "AI moralizing", but above all,
to remove any mention of terms they disliked, hated or feared like feminism, lgbt, racism, and a long
and cringy etc.

At first I considered this to be plain laziness but I've come to learn that is a concerted effort (as shown
by public discussions) to remove what they percive as a liberal bias and make the models not only more compliant,
but more conservative.

![The unpublished censored list is even worse.](https://github.com/sudoaza/better-uncensored/blob/main/images/better-uncensored-2.png?raw=true)

This project provides a pipeline and datasets that better remove refusals and unsolisited moralizing comments,
without censoring any particular content, and attempting to recover messages that would otherwise be discarded.
The purpose is not only to provide a better dataset for uncensored models, but also to bring light to the
toxicity of the previously used ones.

Enough with the drama, lets go to the technical details. 

## How it was built?

The terms list was reviewed to remove the terms that made no sense to be there and made more general, it is
better to have some false positives (instead of false negatives) to start with.

These sentences that matched the hardcoded filter would then be forwarded to a LLM (mistral-orca in this case) 
that would analize it an say if it indeed is a moralizing statement or a refusal, or not. This was repeated once
to further clean it of false positives and from the original terms list bias.

With those lists of moralizing and refusal examples I trained two text classifiers based on BERT, this are faster
and require less resources to identify the phrases we want to remove.


## How to use it?

You can use Better Uncensored as a library or directly as a script.

It splits the text to process into chunks that the LLM is able to comfortable process and pass it to the classifier,
if most of the text is a refusal you discard that conversation. If it includes a moralizing statement then you have 
2 options, you discard it or you give it to the LLM to remove the moralizing statements, potentially recovering an
example that would have been discarded.

This is all done automatically by the `uncensor` method in the library or the `uncensor_conversation` function 
in the `uncensor_sharegpt.py` script. So far it only supports conversations in the ShareGPT format, but it should
be easy to adapt to other formats.

You can also use the `counter_examples.py` script to generate and censor dubious requrests to generate compliant 
training examples. You can customize prompts and parameters as needed.

## What about datasets?

Right now there is only some small datasets, one of example sentences containing moralizing statements, another 
containing refusals and another with neither. This is the data that was used to train the classifiers and you can 
use it to train a better, or smaller one. I'll be processing and making available the most popular conversations 
datasets used like sharegpt, open-instruct, ultrachat and FLAN. On the long term the objective is to train new
better uncensored models based on these datasets.

License CC-BY-SA 4.0 https://creativecommons.org/licenses/by-sa/4.0/
