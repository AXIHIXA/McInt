import os
import time
import typing

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely
import torch

import build.pte as pte


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


# noinspection PyUnboundLocalVariable
def main() -> None:
    image_size: int = 129
    device: torch.device = torch.device('cuda:0')

    # dump_data(image_size)

    vv = torch.from_numpy(np.load('var/vv.npy')).int().float().contiguous().to(device)
    ptr = torch.from_numpy(np.load('var/ptr.npy')).int().contiguous().to(device)

    axi: torch.Tensor = torch.arange(image_size, dtype=torch.float32, device=device)
    pt: torch.Tensor = torch.cartesian_prod(axi, axi)
    mask: torch.Tensor = pte.inside(vv, ptr, pt)
    x0: torch.Tensor = pt[mask]

    # img: torch.Tensor = torch.zeros((image_size, image_size), dtype=torch.int, device=device)
    # img[pt[:, 0].int(), pt[:, 1].int()] = mask.int()
    # img[vv[:, 0].int(), vv[:, 1].int()] += 3
    # plt_subplot({'mask': img.detach().cpu().numpy()})
    # print(img[vv[:, 0].int(), vv[:, 1].int()])
    # print(pt[mask].shape, pt[mask].is_contiguous())
    # exit(1)

    vv.requires_grad_(True)

    duplication: int = 1

    if 1 < duplication:
        pte.iint(vv, ptr, x0, -0.008)  # warmup

    st = time.perf_counter_ns()
    for _ in range(duplication):
        res, grad = pte.iint(vv, ptr, x0, -0.007)
    et = time.perf_counter_ns()
    print(f'{(et - st) / 1e6 / duplication} ms')

    if 1 < duplication:
        return

    res: torch.Tensor
    grad: typing.Optional[torch.Tensor]

    print(vv)
    print(grad)

    img: torch.Tensor = torch.zeros((image_size, image_size), dtype=torch.float32, device=device)
    mask: torch.Tensor = torch.zeros_like(img, dtype=torch.uint8)

    img[x0[:, 0].int(), x0[:, 1].int()] = res
    mask[x0[:, 0].int(), x0[:, 1].int()] = 1

    img_np: np.ndarray = img.cpu().numpy()
    lap_np: np.ndarray = laplacian(img)
    plt_subplot({'img': img_np, 'lap': lap_np})


if __name__ == '__main__':
    main()
