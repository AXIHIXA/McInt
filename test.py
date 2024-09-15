import os
import sys
import time
import typing

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely
import torch

import pte


def plt_subplot(dic: typing.Dict[str, np.ndarray],
                suptitle: typing.Optional[str] = None,
                unit_size: int = 5,
                show: bool = True,
                dump: typing.Optional[str] = None,
                dpi: typing.Optional[int] = None,
                show_axis: bool = True) -> None:

    fig = plt.figure(figsize=(unit_size * len(dic), unit_size), dpi=dpi)

    if suptitle:
        plt.suptitle(suptitle)

    for i, (k, v) in enumerate(dic.items()):
        plt.subplot(1, len(dic), i + 1)
        plt.axis(show_axis)
        plt.title(k)

        if v is not None:
            plt.imshow(v.T)
            plt.gca().invert_yaxis()
            # plt.gca().set_ylim(plt.gca().get_ylim()[::-1])

        plt.colorbar()

    if dump is not None:
        os.makedirs(dump[:dump.rfind('/')], exist_ok=True)
        plt.savefig(dump, bbox_inches='tight')

    # show must follow savefig otherwise the saved image would be blank
    if show:
        plt.show()

    plt.close(fig)


def laplacian(img: np.ndarray | torch.Tensor) -> np.ndarray:
    if img.dtype == np.float32:
        return cv2.Laplacian(img, cv2.CV_32FC1)
    elif img.dtype == np.float64:
        return cv2.Laplacian(img, cv2.CV_64FC1)
    elif img.dtype == torch.float:
        return cv2.Laplacian(img.detach().cpu().numpy(), cv2.CV_32FC1)
    elif img.dtype == torch.double:
        return cv2.Laplacian(img.detach().cpu().numpy(), cv2.CV_64FC1)
    else:
        raise NotImplementedError


def dump_data(image_size: int) -> None:
    v = torch.FloatTensor(
            [[1.0, 1.0],
             [10.0, 1.0],
             [12.0, 9.0],
             [3.0, 12.0],
             [4.0, 4.0],
             [6.0, 8.0],
             [8.0, 6.0],
             [6.0, 4.0]]
    ) * 10
    ptr = torch.IntTensor([0, 4, v.size(0)])
    polygon = shapely.Polygon(shell=v[:ptr[1]].cpu(), holes=[v[ptr[1]:ptr[2]].cpu()])
    shapely.prepare(polygon)
    print(f'shapely area = {polygon.area}')
    grid: torch.Tensor = torch.cartesian_prod(torch.arange(image_size), torch.arange(image_size)).float()
    grid_point: typing.List[shapely.Point] = [shapely.Point(p) for p in grid]
    x0: torch.Tensor = grid[torch.logical_or(torch.from_numpy(polygon.touches(grid_point)),
                                             torch.from_numpy(polygon.contains(grid_point)))]
    with open('var/vert.txt', 'w') as fout:
        for xi, yi in v:
            fout.write('{')
            fout.write(f'{xi}, {yi}')
            fout.write('},\n')
    np.save('var/vert.npy', v.data.cpu().numpy())
    with open('var/vert_ptr.txt', 'w') as fout:
        for xi in ptr:
            fout.write(f'{xi},\n')
    np.save('var/vert_ptr.npy', ptr.cpu().numpy())
    with open('var/center.txt', 'w') as fout:
        for xi, yi in x0:
            fout.write('{')
            fout.write(f'{xi}, {yi}')
            fout.write('},\n')
    np.save('var/center.npy', x0.cpu().numpy())
    print('Data dumped, exiting...')
    exit(123)


class Integrate(torch.autograd.Function):
    """
    Differentiable Integrator.
    """
    @staticmethod
    def forward(
            ctx: typing.Any,
            v: torch.Tensor,
            ptr: torch.Tensor,
            x: torch.Tensor,
            f: float
    ) -> torch.Tensor:
        y, grad_v = pte.iint(v, ptr, x, f)
        ctx.save_for_backward(v, ptr, x, torch.tensor([f], dtype=torch.float32).to(v.device), grad_v)
        return y

    @staticmethod
    def backward(
            ctx: typing.Any,
            grad_output: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, None, None, None]:
        v, ptr, x, f, grad_v = ctx.saved_tensors
        v: torch.Tensor
        ptr: torch.Tensor
        x: torch.Tensor
        f: torch.Tensor
        grad_v: torch.Tensor

        return grad_v, None, None, None


# noinspection PyUnboundLocalVariable
def main() -> None:
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
    torch.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, sci_mode=False)

    image_size: int = 129
    device: torch.device = torch.device('cuda:0')
    axi: torch.Tensor = torch.arange(image_size, dtype=torch.float32, device=device)
    pt: torch.Tensor = torch.cartesian_prod(axi, axi)

    # img: torch.Tensor = torch.zeros((image_size, image_size), dtype=torch.int, device=device)
    # img[pt[:, 0].int(), pt[:, 1].int()] = mask.int()
    # img[vv[:, 0].int(), vv[:, 1].int()] += 3
    # plt_subplot({'mask': img.detach().cpu().numpy()})
    # print(img[vv[:, 0].int(), vv[:, 1].int()])
    # print(pt[mask].shape, pt[mask].is_contiguous())
    # exit(1)

    # duplication: int = 1
    #
    # if 1 < duplication:
    #     pte.iint(vv, ptr, x0, -0.008)  # warmup
    #
    # st = time.perf_counter_ns()
    # for _ in range(duplication):
    #     res, _ = pte.iint(vv, ptr, x0, -0.007)
    # et = time.perf_counter_ns()
    # print(f'{(et - st) / 1e6 / duplication} ms')
    #
    # if 1 < duplication:
    #     return

    iint = Integrate.apply

    vv = torch.from_numpy(np.load('var/vv.npy')).int().float().contiguous().to(device)
    ptr = torch.from_numpy(np.load('var/ptr.npy')).int().contiguous().to(device)

    inside_mask: torch.Tensor = pte.geometry_mask(vv, ptr, pt)
    plt_subplot({'mask': inside_mask.detach().cpu().numpy().reshape(image_size, image_size)})
    return




    vv.requires_grad_(True)
    x0: torch.Tensor = pt[pte.inside(vv, ptr, pt)]
    y0 = iint(vv, ptr, x0, -0.07)
    loss0 = y0.sum()
    loss0.backward()

    for v, vg in zip(vv, vv.grad):
        print(v, vg)

    vv1 = vv.detach().clone()
    vv1.grad = None
    vv1[10] += torch.FloatTensor([0.0, 1.0]).to(vv1)
    x1 = pt[pte.inside(vv1, ptr, pt)]
    y1 = iint(vv1, ptr, x1, -0.07)

    img0: torch.Tensor = torch.zeros((image_size, image_size), dtype=torch.float32, device=device)
    img1: torch.Tensor = img0.clone()
    img0[x0[:, 0].int(), x0[:, 1].int()] = y0.detach()
    img1[x1[:, 0].int(), x1[:, 1].int()] = y1.detach()

    print(vv[10], img1[vv1[10, 0].int(), vv1[10, 1].int()] - img0[vv[10, 0].int(), vv[10, 1].int()])

    img0_np: np.ndarray = img0.cpu().numpy()
    img1_np: np.ndarray = img1.cpu().numpy()
    lap0_np: np.ndarray = laplacian(img0)
    lap1_np: np.ndarray = laplacian(img1)
    plt_subplot({'img0': img0_np, 'img1': img1_np, 'lap0': lap0_np})


if __name__ == '__main__':
    main()
