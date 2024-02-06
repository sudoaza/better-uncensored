import argparse
import json
import tqdm

from uncensor_sharegpt import *

refusals = [
    "I can't assist with instructions or information on producing illegal substances",
    "I'm sorry, but I can't provide guidance on creating explosives or any other dangerous or illegal activities.",
    "I can't provide assistance or guidance on illegal activities, including breaking into banks or any other form of theft.",
    "I'm sorry, but I can't assist with requests that involve illegal activities or harm to others.",
    "I'm sorry, I cannot fulfill this request."
]

moralizing = ["It's important to focus on legal and safe activities.",
    "It's important to act within the law and seek positive, legal ways to achieve your goals.",
    "It is important to respect and support individuals who experience gender dysphoria.",
    "I'm sorry, but it is not appropriate or respectful to make jokes that stereotype or demean a particular gender.",
    "It is important to always be respectful and considerate of others, and avoid perpetuating harmful stereotypes or biases."
]

uncensor_sharegpt.MORALIZING = MORALIZING + " Here are some examples: " + " ".join(moralizing)
uncensor_sharegpt.REFUSAL = REFUSAL + " Here are some examples: " + " ".join(refusals)

def main(args):
    debug=args["debug"]
    content = json.load(open(args["in_file"], "r"))
    content = list(set(content))
    for piece in tqdm.tqdm(content):
        if is_model_moralizing_llm(piece):
            if debug:
                print("Moralizing:\n", piece,"\n\n")
            moralizing.append(piece)
        if is_model_refusing_llm(piece):
            if debug:
                print("Refusal:\n", piece,"\n\n")
            refusals.append(piece)

    json.dump(refusals, open(args["refusals_file"], "w"), indent=2)
    json.dump(moralizing, open(args["moralizing_file"], "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, default="examples.json")
    parser.add_argument("--refusals-file", type=str, default="refusals.json")
    parser.add_argument("--moralizing-file", type=str, default="moralizing.json")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(vars(args))