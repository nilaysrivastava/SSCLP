import argparse

"For dataset 1 and dataset 2 parameter are different, you can modify according to your own needs"

def settings():
    parser = argparse.ArgumentParser()

    # public parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed. Default is 0.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers. Default is 0.')

    parser.add_argument('--in_file', default="../data/dataset1/LDA.edgelist",
                        help='Path to data fold. e.g., ../data/LDA.edgelist')

    parser.add_argument('--neg_sample', default="../data/dataset1/no_LDA.edgelist",
                        help='Path to data fold. e.g., ../data/LDA.edgelist')

    parser.add_argument('--task_type', default="LDA", choices=['LDA', 'MDA', 'LMI'],
                        help='Initial prediction task type. Default is LDA.')

    # Training settings
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate. Default is 5e-4.')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate. Default is 0.5.')

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters) Default is 5e-4.')

    parser.add_argument('--batch', type=int, default=25,
                        help='Batch size. Default is 25.')

    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of epochs to train. Default is 80.')

    parser.add_argument('--loss_ratio1', type=float, default=0.1,
                        help='Ratio of self_supervision. Default is 1 (LDA), 0.1 (MDA,LMI)')

    parser.add_argument('--loss_ratio2', type=float, default=0.1,
                        help='Weight for consistency loss')

    # model parameter setting
    parser.add_argument('--dimensions', type=int, default=512,
                        help='dimensions of feature d. Default is 512 (LDA), 1024 (MDA and LMI)')

    parser.add_argument('--hidden1', type=int, default=256,
                        help='Embedding dimension of encoder layer 1 for SSCLP. Default is d/2.')

    parser.add_argument('--hidden2', type=int, default=128,
                        help='Embedding dimension of encoder layer 2 for SSCLP. Default is d/4.')

    parser.add_argument('--decoder1', type=int, default=512,
                        help='Embedding dimension of decoder layer 1 for SSCLP. Default is 512.')

    # New optional parameters
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from saved checkpoint if available.')

    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip training this fold if saved outputs already exist.')

    args = parser.parse_args()

    return args
