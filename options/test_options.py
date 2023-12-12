from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, opt):
        opt = BaseOptions.initialize(self, opt)  # define shared options
