import os
import json
from utils.config_utils import setup_main
from utils.eval import (
    calc_scanrefer_score,
    calc_scan2cap_score,
    calc_scanqa_score,
    calc_sqa3d_score,
    calc_multi3dref_score
)
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

tokenizer = PTBTokenizer()
scorers = [
    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    (Meteor(), "METEOR"),
    (Rouge(), "ROUGE_L"),
    (Cider(), "CIDEr"),
]

def calculate_scores_for_all_tags(config, tags):
    all_results = {}
    for eval_name in tags:
        output_dir = os.path.join(config.output_dir, f"{config.model}_{eval_name}")
        
        predictions = []
        for chunk_idx in range(config.num_chunks):
            predictions_file = os.path.join(output_dir, f"predictions_chunk_{chunk_idx}.json")
            if os.path.exists(predictions_file):
                with open(predictions_file, "r") as f:
                    predictions.extend(json.load(f))
            else:
                raise FileNotFoundError(f"Missing predictions file for chunk {chunk_idx}: {predictions_file}")

        print(f"Loaded {len(predictions)} predictions for evaluation of {eval_name}.")

        val_scores = {}

        if eval_name == 'scanqa':
            val_scores = calc_scanqa_score(predictions, tokenizer, scorers, config)
        elif eval_name == 'scanrefer':
            val_scores = calc_scanrefer_score(predictions, config)
        elif eval_name in ["scan2cap", "scan2cap_location"]:
            val_scores = calc_scan2cap_score(predictions, tokenizer, scorers, config)
        elif eval_name == "sqa3d":
            val_scores = calc_sqa3d_score(predictions, tokenizer, scorers, config)
        elif eval_name == 'multi3dref':
            val_scores = calc_multi3dref_score(predictions, config)
        else:
            raise NotImplementedError(f"Evaluation for {eval_name} is not implemented.")

        print(f"Evaluation Results for {eval_name}:")
        print(json.dumps(val_scores, indent=4))

        all_results[eval_name] = val_scores

    all_results_file = os.path.join(config.output_dir, "all_evaluation_results.json")
    with open(all_results_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"All evaluation results saved to {all_results_file}")

    return all_results


def main():
    config = setup_main()
    tags = config.calculate_score_tag.split('#')
    calculate_scores_for_all_tags(config, tags)


if __name__ == "__main__":
    main()
