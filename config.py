data_path = "./data"

pre_batch_size = 64
pre_learning_rate = 0.001
pre_num_epochs = 10
pre_hidden_sizes = [512, 256, 128]
pre_dropout_rate = 0.2
pre_model_path = "pretrained.pth"

lora_rank = 4
lora_alpha = 6
lora_batch_size = 64
lora_learning_rate = 0.0001
lora_num_epochs = 4
lora_model_path = "finetuned.pth"
lora_mix_old_digits = True
