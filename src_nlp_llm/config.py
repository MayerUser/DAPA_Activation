# config.py

# Hugging Face Configuration
# Replace "YOUR_HUGGINGFACE_TOKEN" with your actual Hugging Face token
HUGGINGFACE_TOKEN = "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" 

BATCH_SIZE = 256 # Batch size for data loader

# BERT PRE-TRAINED MODEL PATH Config
PATH_SST    = "textattack/bert-base-uncased-SST-2"
PATH_MRPC   = "textattack/bert-base-uncased-MRPC"
# PATH_MRPC   = "Intel/bert-base-uncased-mrpc"
PATH_STS    = "textattack/bert-base-uncased-STS-B"
PATH_QQP    = "textattack/bert-base-uncased-QQP"
PATH_MNLI   = "textattack/bert-base-uncased-MNLI"
PATH_QNLI   = "textattack/bert-base-uncased-QNLI"