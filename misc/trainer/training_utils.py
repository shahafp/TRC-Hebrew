import torch


def get_parameters(params, training_layers):
    params_list = [p for n, p in list(params)]
    for param in params_list[:-training_layers]:
        param.requires_grad = False
    return params_list


def save_checkpoint(save_dir, save_path, model, optimizer, valid_loss):
    if save_dir == None:
        return
    save_path = save_dir + save_path

    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, optimizer, device):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    model.to(device)

    return state_dict['valid_loss']
