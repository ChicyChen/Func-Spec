import torch
import logging


def compare_models(model_1, model_2, step_index, before = True):
    """compare two models to check if they have identical weight layer-wise-ly"""
    if before:
        compare_state = 'before'
    else:
        compare_state = 'after'

    logging.info('=========Model check %s each step===========' % compare_state)
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
    if models_differ == 0:
        print('Models match perfectly %s step %s! :)'% (compare_state, step_index))
        logging.info('Models match perfectly %s step %s! :)'% (compare_state, step_index))
    else:
        print('Models DO NOT match perfectly %s step %s :('% (compare_state, step_index))
        logging.info('Models DO NOT match perfectly %s step %s :('% (compare_state, step_index))
    logging.info('=========Model check %s each step===========' % compare_state)


def round_params(model, decimals=4):
    """Round the weights of all parameters in a model to a specified number of decimal places."""
    for param in model.parameters():
        param.data = torch.round(param.data * 10**decimals) / (10**decimals) # torch.floor() is another option


def round_gradients(model, decimals=4):
    """Round the gradients of all parameters in a model to a specified number of decimal places."""
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = torch.round(param.grad.data * 10**decimals) / (10**decimals)


def get_gradient_and_weight(model, grads_encoder1, grads_encoder2, weights_encoder1, weights_encoder2):
    """get gradients of all layers of encoder1 and encoder2 as dictionary{key: layer_name, value: gradient}
       get weights of all layers of encoder1 and encoder2 as dictionary{key: layer_name, value: weight}"""
    for param_tuple in model.named_parameters():
        name, param = param_tuple
        if 'encoder1' in name and 'net.fc' not in name:
            grads_encoder1[name[16:]] = param.grad
            weights_encoder1[name[16:]] = param
        elif 'encoder2' in name and 'net.fc' not in name:
            grads_encoder2[name[16:]] = param.grad
            weights_encoder2[name[16:]] = param


def compute_MAE(grads_encoder1, grads_encoder2, weights_encoder1, weights_encoder2):
    """compute the mean absolute error of each layers' gradient and weight of encoder1 and encoder2,
       and then sum them together"""
    grads_MAE = 0.0
    weights_MAE = 0.0
    for key in weights_encoder1.keys(): # two dict should have the same key value
        grads_diff  = torch.sum(torch.abs(grads_encoder1[key] - grads_encoder2[key]))
        weights_diff = torch.sum(torch.abs(weights_encoder1[key] - weights_encoder2[key]))
        grads_MAE = grads_MAE + grads_diff
        weights_MAE = weights_MAE + weights_diff
    return grads_MAE, weights_MAE