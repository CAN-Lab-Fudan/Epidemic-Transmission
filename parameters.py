import argparse
from typing import Tuple


def parameter_parser():
    params = argparse.ArgumentParser(description="Parameters in train DynamicsGAT.")

    params.add_argument(
        '--file_path',
        help='hdf5 file name to save SQS data',
        type=str,
        default='data',
    )
    params.add_argument(
        '--group-name',
        help='name of network',
        type=str,
        default='Barabasi_Albert_graph',
    )
    params.add_argument(
        '--N',
        help='--size of network',
        type=int,
        default=1000,
    )
    params.add_argument(
        '--k',
        help='--average degree of network',
        type=int,
        default=4,
    )
    params.add_argument(
        '--lambda_range',
        help='--randge of lambda change',
        type=Tuple[float, float],
        default=(0.01, 0.1),
    )
    params.add_argument(
        '--lambda_numbers',
        help='numbers of lambda change in range',
        type=int,
        default=100,
    )

    params.add_argument(
        '--batch_size',
        help='batch size for train loader',
        type=int,
        default=100,
    )
    params.add_argument(
        '--in_features',
        help='number of in features',
        type=int,
        default=1,
    )
    params.add_argument(
        '--f_in_hidden',
        help='Hidden features of encoder layers',
        type=int,
        default=32,
    )
    params.add_argument(
        '--f_att_in',
        help='Input features of the attention layers',
        type=int,
        default=32,
    )
    params.add_argument(
        '--f_att_hidden',
        help='Hidden features of the attention layers',
        type=int,
        default=32,
    )
    params.add_argument(
       '--f_out_in',
        help='Input features of the decoder layers',
        type=int,
        default=32,
    )
    params.add_argument(
        '--f_out_hidden',
        help='Hidden features of the decoder layers',
        type=int,
        default=32,
    )
    params.add_argument(
        '--out_features',
        help='numbers of features in output layers',
        type=int,
        default=1,
    )
    params.add_argument(
        '--epochs',
        help='numbers of epoch',
        type=int,
        default=30,
    )
    params.add_argument(
        '--lr',
        help='learning rate',
        type=float,
        default=0.001,
    )

    params.add_argument(
        '--check_point',
        help='check point to val model',
        type=int,
        default=5,
    )
    params.add_argument(
        '--model_path',
        help='where to save model',
        type=str,
        default='model.pth',
    )
    params.add_argument(
        '--results-path',
        help='where to save results',
        type=str,
        default='Results',
    )


    return params.parse_args()
