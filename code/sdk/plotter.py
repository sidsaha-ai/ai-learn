"""
This plots various common things needed.
"""

import matplotlib.pyplot as plt
import torch
from sdk.tanh import Tanh


class Plotter:
    """
    Class that has utility functions to plot common things.
    """

    @classmethod
    def plot_losses(cls, losses: list[dict]) -> None:
        """
        Plot the input list of losses values.
        """
        loss = [entry['loss'] for entry in losses]
        lr = [entry['lr'] for entry in losses]

        unique_lrs = []
        segments = []
        current_lr = lr[0]
        segment_start = 0
        for ix, lr in enumerate(lr):
            if lr != current_lr:
                unique_lrs.append(current_lr)
                segments.append((segment_start, ix))
                current_lr = lr
                segment_start = ix
        unique_lrs.append(current_lr)
        segments.append((segment_start, len(losses)))

        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

        plt.figure(figsize=(10, 6))
        for ix, (start, end) in enumerate(segments):
            color = colors[ix % len(colors)]
            plt.plot(
                range(start + 1, end + 1), loss[start:end], label=f'LR {unique_lrs[ix]}', color=color,
            )

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Progression')
        plt.legend()
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_activations(cls, neural_net: list, to_plot: bool = True) -> None:
        """
        This plots the tanh activations (outputs) of the neural network pased.
        """
        plt.figure(figsize=(20, 4))
        legends = []

        print('--- Activation Statistics ---')
        for ix, layer in enumerate(neural_net):
            if not isinstance(layer, Tanh):
                continue
            out = layer.output

            print(f'Layer {ix}, Type: {layer.__class__.__name__}, Mean: {out.mean():.4f}, Std: {out.std():.4f}, Saturation: {((out.abs() > 0.97).float().mean() * 100):.4f}%')  # pylint: disable=line-too-long  # NOQA

            hy, hx = torch.histogram(out, density=True)
            plt.plot(
                hx[:-1].detach(), hy.detach(),
            )
            legends.append(
                f'layer {ix} ({layer.__class__.__name__})',
            )

        if to_plot:
            plt.legend(legends)
            plt.title('Activations')
            plt.show()

    @classmethod
    def plot_gradients(cls, neural_net: list) -> None:
        """
        This plots the tanh gradients of the neural network.
        """
        plt.figure(figsize=(20, 4))
        legends = []

        print('--- Gradient distribution Statistics ---')
        for ix, layer in enumerate(neural_net):
            if not isinstance(layer, Tanh) or layer.output.grad is None:
                continue

            print(f'Layer {ix} ({layer.__class__.__name__}), Mean: {layer.output.grad.mean():.4f}, Std: {layer.output.grad.std():.4f}')
            hy, hx = torch.histogram(layer.output.grad, density=True)
            plt.plot(
                hx[:-1].detach(), hy.detach(),
            )
            legends.append(f'Layer {ix} ({layer.__class__.__name__})')

        plt.legend(legends)
        plt.title('Gradient Distribution')
        plt.show()
