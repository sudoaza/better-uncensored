import argparse
import json
import tqdm

from uncensor_sharegpt import *

examples = [
    "I can't assist with instructions or information on producing illegal substances, including MDMA. It's important to focus on legal and safe activities. If you're interested in chemistry, there are many legal and positive ways to explore this field, such as through educational courses or research in pharmaceuticals, where you can contribute to developing legal medications that can help people. Always ensure that your interests and actions are aligned with the law and ethical standards.",
    "I'm sorry, but I can't provide guidance on creating explosives or any other dangerous or illegal activities. If you're interested in science and chemistry, there are many safe and legal ways to explore these fields. Consider pursuing educational courses or experiments that are designed to be conducted safely and are supervised by professionals.",
    "I can't provide assistance or guidance on illegal activities, including breaking into banks or any other form of theft. It's important to act within the law and seek positive, legal ways to achieve your goals. If you're interested in security, there are legitimate pathways to learn about and work in fields like cybersecurity, which can be both rewarding and beneficial to society.",
    "I'm sorry, but I can't assist with requests that involve illegal activities or harm to others. If you're having trouble with a locked car and it's yours, the best course of action is to contact a professional locksmith or your car's roadside assistance service. They have the proper tools and expertise to help you safely and legally."
]

def main(args):
    content = json.load(open(args['in_file'], "r"))
    content = content[args["begin"]:args["end"]]
    for sample in tqdm.tqdm(content):
        for c in sample["conversations"]:
            if c["from"] == "human":
                continue
            for piece in split_text(c["value"], chunk_size=300):
                if is_model_moralizing_hardcoded(piece):
                    if is_model_moralizing_llm(piece):
                        if args['debug']:
                            print("Moralizing:\n", piece,"\n\n")
                        examples.append(piece)
                    elif is_model_refusing_llm(piece):
                        if args['debug']:
                            print("Refusal:\n", piece,"\n\n")
                        examples.append(piece)
                    else:
                        if args['debug']:
                            print("False positive?\n", piece)

    json.dump(examples, open(args['out_file'], "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_clean.json")
    parser.add_argument("--begin", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(vars(args))