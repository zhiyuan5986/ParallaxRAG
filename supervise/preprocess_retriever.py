import os
import sys
import torch

from src.dataset.preprocess.emb import load_model_and_tokenizer
from supervise.utils import build_processed_pickles
from supervise.preprocess_embeddings import compute_and_save_embeddings


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess retriever data and embeddings (webqsp/cwq/kqapro/metaqa)')
    parser.add_argument('--dataset_name', '-d', type=str, default='webqsp',
                        choices=['webqsp', 'cwq', 'kqapro', 'metaqa', 'bioasq'],
                        help='dataset to preprocess')
    parser.add_argument('--build_pkl', action='store_true', 
                        help='build processed pickles from HuggingFace dataset (for webqsp/cwq/kqapro/metaqa)')
    args = parser.parse_args()

    dataset_name = args.dataset_name.lower()

    # 1) processed files (entity align + ids)
    if dataset_name == 'bioasq':
        base_dir = "/data/BioASQ"
        proc_dir = os.path.join(base_dir, 'proc_resolved')
        processed_train = os.path.join(proc_dir, 'train.json')
        processed_val = None  # BioASQ has no validation split
        processed_test = os.path.join(proc_dir, 'test.json')
        print(f"[INFO] Using BioASQ processed files: train={processed_train}, test={processed_test}")
        print(f"[INFO] Note: BioASQ has no validation set")
    else:
        base_dir = f"/data/{dataset_name}"
        processed_dir = os.path.join(base_dir, 'processed')
        # Build processed pickles from HuggingFace if requested
        if args.build_pkl:
            print(f"[INFO] Building processed pickles for {dataset_name} from HuggingFace...")
            build_processed_pickles(processed_dir, dataset_name=dataset_name)
        processed_train = os.path.join(processed_dir, 'train.pkl')
        processed_val = os.path.join(processed_dir, 'val.pkl')
        processed_test = os.path.join(processed_dir, 'test.pkl')

    # 2) Load BGE-M3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model_and_tokenizer()
    model.to(device)
    model.eval()

    # 3) Encode and save emb dicts in original layout

    emb_root = os.path.join(base_dir, 'emb', 'bge')
    os.makedirs(emb_root, exist_ok=True)

    # encode available splits
    if processed_train and os.path.exists(processed_train):
        compute_and_save_embeddings(processed_train, os.path.join(emb_root, 'train.pth'), model, tokenizer, device)
    else:
        print(f"[WARN] processed train file not found: {processed_train}, skip.")
    if processed_val and os.path.exists(processed_val):
        compute_and_save_embeddings(processed_val, os.path.join(emb_root, 'val.pth'), model, tokenizer, device)
    else:
        print(f"[INFO] processed val file not found or not applicable for this dataset, skip.")
    if processed_test and os.path.exists(processed_test):
        compute_and_save_embeddings(processed_test, os.path.join(emb_root, 'test.pth'), model, tokenizer, device)
    else:
        print(f"[WARN] processed test file not found: {processed_test}, skip.")

    print(' Preprocess done. Files ready for retriever training:')
    present_train = os.path.exists(processed_train) if processed_train else False
    present_val = (processed_val and os.path.exists(processed_val)) if processed_val else False
    present_test = os.path.exists(processed_test) if processed_test else False
    print(f" - processed (input): train={'OK' if present_train else 'N/A'}, val={'OK' if present_val else 'N/A'}, test={'OK' if present_test else 'N/A'}")
    print(f" - embeddings (output): {base_dir}/emb/bge/{{train,val,test}}.pth")


if __name__ == '__main__':
    main()
