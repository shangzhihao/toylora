import argparse

from finetune import finetune
from inference import inference
from pretrain import pretrain


jobs = {"pretrain", "finetune", "inference"}


def main(job="pretrain"):
    if job not in jobs:
        job = "pretrain"
        print(f"Invalid job '{job}'. Using 'pretrain' instead.")
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
    parser = argparse.ArgumentParser(
        description="ToyLoRA: Educational LoRA demonstration"
    )
    parser.add_argument(
        "--job",
        type=str,
        default="pretrain",
        choices=["pretrain", "finetune", "inference"],
        help="Job to run: pretrain (train base model on digits 0-7), finetune (LoRA on digits 8-9), or inference (test model)",
    )

    args = parser.parse_args()
    main(args.job)
