import torch


class EMA:
    def __init__(self, model, decay):
        if not 0.0 < decay < 1.0:
            raise ValueError("Decay must be between 0 and 1.")
        self.model = model
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update(self):
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self.shadow[name].lerp_(parameter, 1.0 - self.decay)

    def apply_shadow(self):
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(self.shadow[name])
