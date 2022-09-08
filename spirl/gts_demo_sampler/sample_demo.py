


class SampleDemo:
    def __init__(self, args):
        self.args = args
        
        # set up params
        self.conf = conf = self.get_config()

        self._hp = self._default_hparams()