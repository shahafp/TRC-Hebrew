def get_parameters(params, training_layers):
    params_list = [p for n, p in list(params)]
    for param in params_list[:-training_layers]:
        param.requires_grad = False
    return params_list
