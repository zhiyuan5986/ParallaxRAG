HARDCODED_CONFIG = {
    'env': {
        'num_threads': 16,
        'seed': 42,
    },
    'dataset': {
        'name': 'webqsp',               # used by RetrieverDataset
        'text_encoder_name': 'bge',     # must match preprocess output dir: data_files/webqsp/emb/bge
    },
    'retriever': {
        'topic_pe': True,
        'DE_kwargs': {
            'num_rounds': 2,
            'num_reverse_rounds': 2, 
            'vendi_beta': 0.5,
        },
        'num_heads': None,
        'head_dim': None,
    },
    'optimizer': {
        'lr': 1e-3,
    },
    'eval': {
        'k_list': [100],
    },
    'train': {
        'num_epochs': 100,
        'patience': 20,
        'save_prefix': 'webqsp',
        'warmup_epochs': 0,

        'listwise_warmup_epochs': 3,
        'gate_topk': 16,
        'eval_use_gate': True,
        'gradient_accumulation_steps': 2,

        'log_every': 0,
    },
}
