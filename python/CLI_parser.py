from argparse import ArgumentParser


class CLI_Parser(object):
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            help="choose dataset: \n\t- 'cifar10'\n\t- 'cifar100'",
        )
        parser.add_argument(
            "--LRDecay",
            type=bool,
            required=True,
            help="specify learning rate decay type:\n\t- 'cosine'\n\t- 'linear'",
        )
        parser.add_argument(
            "--probes", type=bool, required=False, default=True, help="add probes"
        )
        parser.add_argument(
            "--kahan", type=bool, required=False, default=True, help="add probes"
        )
        self.args = parser.parse_args()

    # return each arguments values
    def __call__(self):
        return self.args
