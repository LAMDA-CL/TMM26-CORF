def get_model(model_name, args):
    name = model_name.lower()
    if name == "icarl":
        from models.icarl import iCaRL
        return iCaRL(args)
    elif name == "icarl_corf":
        from models.icarl_corf import Learner
        return Learner(args)
    else:
        assert 0
