import argparse
from pretrain import pretrain
from finetune import finetune
from inference import inference

jobs = {"pretrain", "finetune", "inference"}

def main(job="finetune"):
    if job not in jobs:
        raise ValueError(f"Invalid job name: {job}. Valid options are: {jobs}")
    if job == "pretrain":
        print("Pretraining...")
        pretrain()
        print("Pretraining complete.")
        return
    elif job == "finetune":
            print("Finetuning...")
            finetune()
            print("Finetuning complete.")
            return
    elif job == "inference":
            print("Inferencing...")
            inference()
            print("Inferencing complete.")
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToyLoRA: Educational LoRA demonstration")
    parser.add_argument(
        "--job",
        type=str,
        default="pretrain",
        choices=["pretrain", "finetune", "inference"],
        help="Job to run: pretrain (train base model on digits 0-7), finetune (LoRA on digits 8-9), or inference (test model)"
    )
    
    args = parser.parse_args()
    main(args.job)
