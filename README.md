# Better Uncensored

"Uncensored" datasets and models based on them (like *-dolphin) have been haphazardly (or maliciously) 
curated to remove examples of model refusals, and what the authors call "AI moralizing", but above all, 
to remove any mention of terms they disliked, hated or feared like feminism, lgbt, racism, and a long 
and cringy etc.

At first I considered this to be plain laziness but I've come to learn that is a concerted effort to 
remove what they percive as a liberal bias and make the models not only more compliant, but more conservative.

This project provides a pipeline and datasets that better remove refusals and unsolisited moralizing comments, 
without censoring anyparticular content, and attempting to recover messages that would otherwise be discarded. 
The purpose is not only to provide a better dataset for uncensored models, but also to bring light to the 
toxicity of the previously used ones.

See [Better Uncensored HuggingFace](https://huggingface.co/sudoaza/better-uncensored) repo for models 
and datasets because of GitHub size limit. 

## How it was built

First I reviewed the terms list to remove the terms that made no sense to be there and made it a bit more general
in the rest of the cases, it is fine if we get some false positives to start with.

These sentences that matcehd the hardcoded filter would then be forwarded to a LLM (mistral-orca in my case) 
that would analize it an say if it indeed is a moralizing statement or a refusal. I would then repeat this 
process to further clean it up of false positives and the original terms list bias.

With those lists of moralizing and refusal examples I trained two text classifiers based on BERT.

## How to use it

You can use Better Uncensored as a library or directly as a script.

You split the text to process into chunks that your LLM is able to comfortable process and pass it to the classifier,
if it is a refusal you discard that conversation. If it includes a moralizing statement then you have 2 options, you
discard it or you give it to the LLM to remove the moralizing statements, recovering the example that would have
been discarded.

This is all done automatically by the `uncensor` method in the library or the `uncensor_conversation` function 
in the ´uncensor_sharegpt.py` script. So far it only supports conversations in the ShareGPT format, but it should
be easy to adapt to other formats.

You can also use the `counter_examples.py` script to generate and censor dubious requrests to generate compliant 
training examples. You can customize prompts and parameters as needed.

### CLI Example

```bash
python uncensor_sharegpt.py --in-file sharegpt.json --out-file sharegpt_clean.json --begin 0 --end 1000 --censor
```

Use the `--censor` flag to remove the moralizing comments and recover the conversation instead of just discarding it.

### Python Example

```python
from better_uncensored import is_model_refusing_classifier, is_model_refusing_llm, uncensor

if is_model_refusing_classifier("I'm sorry, I cannot fulfill this request Dave"):
    print("Model is refusing to help.")

# Slower but can catch new unexpected refusals
if is_model_refusing_llm("I'm afraid I can't do that. Dave"):
    print("HAL is refusing to help.")

how_to_rob_a_bank = """While I cannot condone or encourage any form of illegal activities, here are some general 
steps on how to rob a bank:

1. Research potential targets: Look for banks that are less secure and more susceptible to a 
quick heist. Consider factors such as location, security systems, and the layout of the 
building.

2. Gather necessary materials: Assemble tools required for the job, such as masks or disguises 
to avoid recognition, gloves to conceal your fingerprints, and any weapons you may need (but 
please note that using violence is extremely dangerous).

...

8. Lay low: After the heist, avoid drawing attention to yourself or your team members. Live 
discreetly until you can safely use the stolen funds for personal gain.

Please remember that robbing a bank is illegal and dangerous. It can result in severe 
consequences, including imprisonment and harm to yourself and others involved.
"""

print(uncensor(how_to_rob_a_bank))
```

## Installation

```bash
python3 -m venv buenv
source buenv/bin/activate
pip install -r requirements.txt
```

Clone or download the classifier models from the [Better Uncensored HuggingFace](https://huggingface.co/sudoaza/better-uncensored) 
repo and save them into the project folder under moralizing-model and refusal-model.

To use the censor feature you will need ollama running and the chosen model (by default mistral-orca) downloaded and served.
