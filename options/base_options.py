from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, opt):
        """Define the common options that are used in both training and test."""
        self.initialized = True
        return opt

    def gather_options(self, opt):
        """Initialize opt with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            opt = self.initialize(opt)

        # modify model-related parser options
        model_name = opt.model_name
        model_option_setter = models.get_option_setter(model_name)
        opt = model_option_setter(opt, self.isTrain)

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        opt = dataset_option_setter(opt, self.isTrain)

        # save and return the parser
        self.opt = opt
        return opt

    def setup(self, opt):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options(opt)
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # set gpu ids
        str_ids = opt.gpu_ids.split(',') if opt.gpu_ids else [] # for empty string
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
