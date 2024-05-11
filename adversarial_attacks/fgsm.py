"""
Fast Gradient Sign Method (FGSM) for generating adversarial examples
in a regression task.
"""

import torch


def FGSM(model, obs, target_action, epsilon, device = 'cpu'):
    """
    Fast Gradient Sign Method (FGSM) for generating adversarial examples
    in a regression task.

    Parameters
    ----------
    model : torch.nn.Module
        The model to attack.
    x : torch.Tensor
        The input tensor.
    y : torch.Tensor
        The target tensor.
    epsilon : float
        The perturbation size.

    Returns
    -------
    torch.Tensor
        The adversarial example.
    """
    # Extract the image input from the observation
    encoder = model.nets["encoder"].nets["obs"].obs_nets['agentview_image']
    # print(encoder, type(encoder))
    obs_nets = model.nets["encoder"].nets["obs"].obs_nets
    # create an empty tensor
    other_modalities = []
    for key, _ in obs_nets.items():
        if key != 'agentview_image':
            # put all the values in the array in the list
            for i in range(len(obs[key])):
                other_modalities.append(obs[key][i])
    # convert the other_modalities to a tensor
    other_modalities = torch.tensor(other_modalities).float().unsqueeze(0).to(device)
    mlp = model.nets["mlp"]
    # print(mlp, type(mlp))
    decoder = model.nets["decoder"]
    # print(decoder, type(decoder))

    # create a network based on the encoder, mlp, and decoder
    model = torch.nn.ModuleDict(
        {
            "encoder": encoder,
            "mlp": mlp,
            "decoder": decoder
        }
    )

    image_input = obs['agentview_image']
    # convert image to tensor
    image_tensor = torch.from_numpy(image_input).float()
    # increase the first dimension
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Create a tensor with the same shape as the image and requires gradients
    perturbed_image = image_tensor.clone().detach().requires_grad_(True)

    # pass the perturbed image through encoder
    encoded_image = model["encoder"](perturbed_image)
    # concatenate the encoded_image with other values of the dict to form a tensor
    encoded_tensor = torch.cat((encoded_image, other_modalities), dim=1)
    # pass the encoded tensor through MLP and decoder
    predicted_action = model["decoder"](model["mlp"](encoded_tensor))['action']
    # convert target action into tensor with shape similar to predicted_action
    target_action = torch.tensor(target_action).float().unsqueeze(0).to(device)
    # print(predicted_action.shape, type(target_action))


    # Compute the MSE loss between the predicted action and the target action
    loss = torch.nn.functional.mse_loss(predicted_action, target_action)

    # Clear the gradients of all parameters
    model.zero_grad()

    # Compute the gradients of the loss with respect to the perturbed image only
    loss.backward(retain_graph=True)

    # Extract the gradients of the perturbed image
    image_grad = perturbed_image.grad.data

    # Compute the sign of the gradients
    sign_data_grad = image_grad.sign()

    # Create the perturbed image by adding the sign of the gradients multiplied by epsilon
    perturbed_image = perturbed_image + epsilon * sign_data_grad

    # Clamp the perturbed image to ensure it remains within the valid range
    perturbed_image = torch.clamp(perturbed_image, 0, 1).squeeze(0)
    # calculate the l2 distance between the two images
    l2_distance = torch.norm(perturbed_image - image_tensor)
    # print("L2 distance between the original and perturbed image: ", l2_distance)

    perturbed_obs = obs
    # Update the observation dictionary with the perturbed image
    perturbed_obs['agentview_image'] = perturbed_image.detach()

    return perturbed_obs

def FGSM_RNN(model, obs, target_action, epsilon, device = 'cpu'):
    """
    Fast Gradient Sign Method (FGSM) for generating adversarial examples
    in a regression task.

    Parameters
    ----------
    model : torch.nn.Module
        The model to attack.
    x : torch.Tensor
        The input tensor.
    y : torch.Tensor
        The target tensor.
    epsilon : float
        The perturbation size.

    Returns
    -------
    torch.Tensor
        The adversarial example.
    """
    # Extract the image input from the observation
    print(model)
    encoder = model.nets["encoder"].nets["obs"].obs_nets['agentview_image']
