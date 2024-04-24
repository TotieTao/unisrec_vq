import logging


def check_param_load(checkpoint, model_param, prefix):
    logging.info("Check parameter loading in model: {}".format(prefix))
    load_keys = set(checkpoint.keys())
    logging.info("load_keys: {}".format(load_keys))
    model_keys = set(model_param.keys())
    logging.info("model_keys: {}".format(model_keys))

    logging.info("Keys in loaded but not in model: {}".format(load_keys.difference(model_keys)))
    logging.info("Keys in model but not in loaded: {}".format(model_keys.difference(load_keys)))