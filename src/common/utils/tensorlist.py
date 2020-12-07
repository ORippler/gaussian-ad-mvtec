"""TensorList class that acts like a (stacked) Tensor."""

from collections import Iterable, UserList, OrderedDict, namedtuple

import torch


class TensorList(UserList):
    """A list of torch Tensors that acts like a (stacked) Tensor.

    This class forwards a bunch of torch methods to their child elements.
    It acts like torch.stack([...], dim=0), but works with tensors of different
    shapes.
    However all Tensors must have the same number of dimensions and be stored
    on the same device.
    (This invariant may be broken between calls for subsequent updates of the
    contained Tensors but must hold on each call to TensorList).
    """

    # List methods.

    def __init__(self, initlist=None):
        super().__init__(initlist)
        for item in self:
            assert isinstance(item, torch.Tensor) or isinstance(
                item, TensorList
            ), "All items of TensorList must be Tensor or TensorList."
            assert item.dim() == self[0].dim(), (
                "All items of TensorList must have the same number of "
                "dimensions."
            )
            assert (
                item.device == self[0].device
            ), "All items of TensorList must be stored on the same device."

    def append(self, item):
        assert isinstance(item, torch.Tensor) or isinstance(
            item, TensorList
        ), "Appended item must be Tensor or TensorList."
        if self:
            assert (
                item.dim() == self[0].dim()
            ), "Appended item has wrong dimension: {} != {}".format(
                self.dim, self[0].dim()
            )
        super().append(item)

    def insert(self, i, item):
        assert isinstance(item, torch.Tensor) or isinstance(
            item, TensorList
        ), "Inserted item must be Tensor or TensorList."
        if self:
            assert (
                item.dim() == self[0].dim()
            ), "Inserted item has wrong dimension: {} != {}".format(
                self.dim, self[0].dim()
            )
        super().insert(i, item)

    # TODO __getitem__(self, *indices)

    # Override list __add__ so it does not append (mimic Tensor).
    def __add__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a + b for a, b in zip(self, other)])
        else:  # Assume scalar.
            return self.__class__([a + other for a in self])

    def __radd__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a + b for a, b in zip(other, self)])
        else:  # Assume scalar.
            return self.__class__([other + a for a in self])

    def __iadd__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            for item, o in zip(self, other):
                item += o
        else:  # Assume scalar.
            for item in self:
                item += other
        return self

    # Override __neg__ to mimic Tensor.
    def __neg__(self):
        return self.__class__([-item for item in self])

    # Override __sub__ to mimic Tensor.
    def __sub__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a - b for a, b in zip(self, other)])
        else:  # Assume scalar.
            return self.__class__([a - other for a in self])

    def __rsub__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a - b for a, b in zip(other, self)])
        else:  # Assume scalar.
            return self.__class__([other - a for a in self])

    def __isub__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            for item, o in zip(self, other):
                item -= o
        else:  # Assume scalar.
            for item in self:
                item -= other
        return self

    # Override list __mul__ so it does not append (mimic Tensor).
    def __mul__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a * b for a, b in zip(self, other)])
        else:  # Assume scalar.
            return self.__class__([a * other for a in self])

    def __rmul__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a * b for a, b in zip(other, self)])
        else:  # Assume scalar.
            return self.__class__([other * a for a in self])

    def __imul__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            for item, o in zip(self, other):
                item *= o
        else:  # Assume scalar.
            for item in self:
                item *= other
        return self

    # Override __floordiv__ to mimic Tensor.
    def __floordiv__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a // b for a, b in zip(self, other)])
        else:  # Assume scalar.
            return self.__class__([a // other for a in self])

    def __rfloordiv__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a // b for a, b in zip(other, self)])
        else:  # Assume scalar.
            return self.__class__([other // a for a in self])

    def __ifloordiv__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            for item, o in zip(self, other):
                item //= o
        else:  # Assume scalar.
            for item in self:
                item //= other
        return self

    # Override __truediv__ to mimic Tensor.
    def __truediv__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a / b for a, b in zip(self, other)])
        else:  # Assume scalar.
            return self.__class__([a / other for a in self])

    def __rtruediv__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a / b for a, b in zip(other, self)])
        else:  # Assume scalar.
            return self.__class__([other / a for a in self])

    def __itruediv__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            for item, o in zip(self, other):
                item /= o
        else:  # Assume scalar.
            for item in self:
                item /= other
        return self

    # Override __mod__ to mimic Tensor.
    def __mod__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a % b for a, b in zip(self, other)])
        else:  # Assume scalar.
            return self.__class__([a % other for a in self])

    def __rmod__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a % b for a, b in zip(other, self)])
        else:  # Assume scalar.
            return self.__class__([other % a for a in self])

    def __imod__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            for item, o in zip(self, other):
                item %= o
        else:  # Assume scalar.
            for item in self:
                item %= other
        return self

    # Override __pow__ to mimic Tensor.
    def __pow__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a ** b for a, b in zip(self, other)])
        else:  # Assume scalar.
            return self.__class__([a ** other for a in self])

    def __rpow__(self, other):
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            return self.__class__([a ** b for a, b in zip(other, self)])
        else:  # Assume scalar.
            return self.__class__([other ** a for a in self])

    def __ipow__(self, other):
        raise NotImplementedError("In-place pow not implemented in torch.")

    # Torch methods (only the useful ones for now).

    def to_tensor(self):  # TODO or is it tensor()
        """Return a tensor of the contents of this list.

        This will fail if the shapes or devices do not line up.
        """
        if not self:
            return torch.Tensor()
        if isinstance(self[0], TensorList):
            return torch.stack([item.to_tensor() for item in self], dim=0)
        return torch.stack(self.data, dim=0)  # TODO nicer error handling.

    @property
    def is_cuda(self):
        # Empty TensorLists are on cpu but move to gpu instantly.
        return self and self[0].is_cuda

    @property
    def device(self):
        return self[0].device if self else torch.device("cpu")

    @property
    def grad(self):
        return [item.grad for item in self]
        # TODO grad setter? is this useful?

    @property
    def requires_grad(self):
        return self[0].requires_grad if self else False

    @requires_grad.setter
    def requires_grad(self, value):
        for item in self:
            item.requires_grad = value

    def abs(self):
        """See :func:`~torch.abs`."""
        return self.__class__([item.abs() for item in self])

    def abs_(self):
        """In-place version of :func:`~TensorList.abs`."""
        for item in self:
            item.abs_()
        return self

    def cpu(self):
        """See :func:`~torch.Tensor.cpu`."""
        return self.__class__([item.cpu() for item in self])

    def cuda(self, device=None, non_blocking=False):
        """See :func:`~torch.Tensor.cuda`."""
        return self.__class__(
            [
                item.cuda(device=device, non_blocking=non_blocking)
                for item in self
            ]
        )

    def detach(self):
        """See :func:`~torch.Tensor.detach`."""
        return self.__class__([item.detach() for item in self])

    def dim(self):
        """See :func:`~torch.Tensor.dim`."""
        return (
            self[0].dim() + 1 if self else 1
        )  # Matching torch.Tensor().dim().

    def listdim(self):
        """Number of dimensions that are actually TensorLists.

        This is always less than self.dim().
        """
        if not self:
            return 1  # Matching dim().
        if isinstance(self[0], TensorList):
            return self[0].listdim() + 1
        else:
            return 1

    def exp(self):
        """See :func:`~torch.Tensor.exp`."""
        return self.__class__([item.exp() for item in self])

    def exp_(self):
        """In-place version of :func:`~TensorList.exp`."""
        for item in self:
            item.exp_()
        return self

    def flatten(self, start_dim=0, end_dim=-1):
        """See :func:`~torch.flatten`."""
        # Convert to positive indices first.
        if start_dim < 0:
            start_dim += self.dim()
        if end_dim < 0:
            end_dim += self.dim()

        if end_dim > 1:  # Only pass to children if needed.
            result = [
                item.flatten(
                    start_dim=start_dim - 1 if start_dim > 0 else start_dim,
                    end_dim=end_dim - 1,
                )
                for item in self
            ]
        else:
            result = self.data  # Share data similar to torch flatten.
        if start_dim == 0 and end_dim > 0:
            # Flatten self into Tensor.
            # This may fail if the Tensors are not of the same shape.
            if result and isinstance(result[0], TensorList):
                result = [item.to_tensor() for item in result]
            return torch.cat(result, dim=0)
        else:
            return self.__class__(result)

    def get_device(self):
        """See :func:`~torch.Tensor.get_device`."""
        if not self:
            raise ValueError("Cannot get device id for empty TensorList.")
        return self[0].get_device()

    def item(self):
        """See :func:`~torch.Tensor.item`."""
        if len(self) != 1:
            raise ValueError(
                "only one element TensorLists can be converted to"
                " python scalars"
            )
        return self[0].item()

    def log(self):
        """See :func:`~torch.Tensor.log`."""
        return self.__class__([item.log() for item in self])

    def log_(self):
        """In-place version of :func:`~TensorList.log`."""
        for item in self:
            item.log_()
        return self

    MaxResult = namedtuple("MaxResult", ["values", "indices"])

    def max(self, dim=None, keepdim=False):
        """See :func:`~torch.max`."""
        if not self:
            raise ValueError("Cannot perform max on an empty TensorList.")
        if dim is None:
            result = max(item.max(keepdim=False) for item in self)
            if keepdim:
                return result.expand([1 for i in range(self.dim())])
            else:
                return result
        # Return a torch.return_types.max tuple of values & indices.
        if dim == 0 or dim == -self.dim():
            if isinstance(self[0], TensorList):
                values_indices = [
                    torch.stack([item[l] for item in self], dim=0).max(dim=0)
                    for l in range(len(self[0]))
                ]
                values = self.__class__([vi.values for vi in values_indices])
                indices = self.__class__([vi.indices for vi in values_indices])
                if keepdim:
                    return self.MaxResult(
                        self.__class__([values]), self.__class__([indices])
                    )
                else:
                    return self.MaxResult(values, indices)
            else:
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().max(dim=0, keepdim=keepdim)
        else:
            values_indices = [
                item.max(dim=dim - 1 if dim > 0 else dim, keepdim=keepdim)
                for item in self
            ]
            values = self.__class__([vi.values for vi in values_indices])
            indices = self.__class__([vi.indices for vi in values_indices])
            return self.MaxResult(values, indices)

    def mean(self, dim=None, keepdim=False):
        """See :func:`~torch.mean`."""
        if not self:
            if keepdim:
                return torch.Tensor([float("nan")], device=self.device).expand(
                    [1 for i in range(self.dim())]
                )
            else:
                return torch.Tensor([float("nan")], device=self.device)
        if isinstance(dim, Iterable):  # Fix broken iterables.
            if not dim:
                dim = None
            elif len(dim) == 1:
                dim = dim[0]

        if dim is None:
            result = sum(item.sum() for item in self) / self.numel()
            if keepdim:
                return result.expand([1 for i in range(self.dim())])
            else:
                return result
        # Check for dim 0 first.
        if isinstance(dim, Iterable):
            if 0 in dim or -self.dim() in dim:
                # The case where the contained item is TensorList must
                # have the same shape and so it returns a full tensor.
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().mean(dim=dim, keepdim=keepdim)
        elif dim == 0 or dim == -self.dim():
            if isinstance(self[0], TensorList):
                result = self.__class__(
                    [
                        sum(item[l] for item in self) / len(self)
                        for l in range(len(self[0]))
                    ]
                )
                if keepdim:
                    return self.__class__([result])
                else:
                    return result
            else:
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().mean(dim=0, keepdim=keepdim)

        # No dim 0 present.
        if isinstance(dim, Iterable):
            dim = [(d - 1 if d > 0 else d) for d in dim]
        else:
            dim = dim - 1 if dim > 0 else dim
        return self.__class__(
            [item.mean(dim=dim, keepdim=keepdim) for item in self]
        )

    MinResult = namedtuple("MinResult", ["values", "indices"])

    def min(self, dim=None, keepdim=False):
        """See :func:`~torch.min`."""
        if not self:
            raise ValueError("Cannot perform min on an empty TensorList.")
        if dim is None:
            result = min(item.min(keepdim=False) for item in self)
            if keepdim:
                return result.expand([1 for i in range(self.dim())])
            else:
                return result
        # Return a torch.return_types.min tuple of values & indices.
        if dim == 0 or dim == -self.dim():
            if isinstance(self[0], TensorList):
                values_indices = [
                    torch.stack([item[l] for item in self], dim=0).min(dim=0)
                    for l in range(len(self[0]))
                ]
                values = self.__class__([vi.values for vi in values_indices])
                indices = self.__class__([vi.indices for vi in values_indices])
                if keepdim:
                    return self.MinResult(
                        self.__class__([values]), self.__class__([indices])
                    )
                else:
                    return self.MinResult(values, indices)
            else:
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().min(dim=0, keepdim=keepdim)
        else:
            values_indices = [
                item.min(dim=dim - 1 if dim > 0 else dim, keepdim=keepdim)
                for item in self
            ]
            values = self.__class__([vi.values for vi in values_indices])
            indices = self.__class__([vi.indices for vi in values_indices])
            return self.MinResult(values, indices)

    def numel(self):
        """See :func:`~torch.numel`."""
        return sum(item.numel() for item in self)

    def numpy(self):
        """See :func:`~torch.Tensor.numpy`."""
        # This may fail if the Tensors are not of the same shape.
        return self.to_tensor().numpy()

    def permute(self, *dims):
        """See :func:`~torch.Tensor.permute`."""
        if len(dims) != self.dims():
            raise ValueError("Number of dims does not match in permute.")
        if dims[0] == 0 or dims[0] == -self.dims():
            dims = ((d - 1 if d > 0 else d) for d in dims[1:])
            return self.__class__([item.permute(*dims) for item in self])
        elif (
            self
            and isinstance(self[0], TensorList)
            and (dims[1] == 0 or dims[1] == -self.dims())
            and (dims[0] == 1 or dims[0] == 1 - self.dims())
        ):
            dims = ((d - 2 if d > 0 else d) for d in dims[2:])
            # TensorList permutation only implemented for two levels.
            return self.__class__(
                [
                    self.__class__(
                        [self[i][j].permute(*dims) for i in range(len(self))]
                    )
                    for j in range(len(self[0]))
                ]
            )
        else:
            # This may fail if the Tensors are not of the same shape.
            return self.to_tensor().permute(*dims)

    def pow(self, exponent):
        """See :func:`~torch.pow`."""
        # Forward to dunder method.
        return self ** exponent

    def pow_(self, exponent):
        """In-place version of :func:`~TensorList.pow`."""
        if isinstance(other, TensorList) or (
            isinstance(other, torch.Tensor) and other.numel() > 1
        ):
            # This does not implement torch's broadcasting rules.
            for item, o in zip(self, other):
                item.pow_(o)
        else:  # Assume scalar.
            for item in self:
                item.pow_(other)
        return self

    def prod(self, dim=None, keepdim=False, dtype=None):
        """See :func:`~torch.prod`."""
        if not self:
            if keepdim:
                return (
                    torch.Tensor([1])
                    .to(dtype=dtype)
                    .expand([1 for i in range(self.dim())])
                )
            else:
                return torch.Tensor([1]).to(dtype=dtype)

        if dim is None:
            result = torch.Tensor([1], device=self.device)
            for item in self:
                result *= item.prod(dtype=dtype)
            if keepdim:
                return result.expand([1 for i in range(self.dim())])
            else:
                return result
        # Check for dim 0 first.
        if dim == 0 or dim == -self.dim():
            if isinstance(self[0], TensorList):
                result = [None] * len(self[0])
                for l in range(len(self[0])):
                    result[l] = self[0][l].to(dtype=dtype)
                    if len(self) > 1:
                        for item in self[1:]:
                            result[l] = result[l] * item[l].to(dtype=dtype)
                result = self.__class__(result)
                if keepdim:
                    return self.__class__([result])
                else:
                    return result
            else:
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().prod(
                    dim=0, keepdim=keepdim, dtype=dtype
                )
        # dim != 0:
        dim = dim - 1 if dim > 0 else dim
        return self.__class__(
            [item.prod(dim=dim, keepdim=keepdim, dtype=dtype) for item in self]
        )

    def requires_grad_(self, requires_grad=True):
        """See :func:`~torch.Tensor.requires_grad_`."""
        for item in self:
            item.requires_grad_(requires_grad)
        return self

    def reshape(self, *shape):
        """See :func:`~torch.Tensor.reshape`."""
        # By its definition reshape requires contiguous storage.
        # While it would be possible to come up with an alternative API that
        # allows specifying sub-shapes, that is very hard to understand.
        return self.flatten().reshape(*shape)

    def sqrt(self):
        """See :func:`~torch.sqrt`."""
        return self.__class__([item.sqrt() for item in self])

    def sqrt_(self):
        """In-place version of :func:`~TensorList.sqrt`."""
        for item in self:
            item.sqrt_()
        return self

    def squeeze(self, dim=None):
        """See :func:`~torch.squeeze`."""
        if len(self) == 0:
            return torch.Tensor()
        if dim == 0 or dim == -self.dim():
            # Try to squeeze self.
            if len(self) == 1:
                return self[0]
            return self
        elif dim is None:
            # Try to squeeze self.
            if len(self) == 1:
                return self[0].squeeze()
            return TensorList([item.squeeze() for item in self])
        if dim < 0:
            dim += self.dim()
        return TensorList([item.squeeze(dim - 1) for item in self])

    def std(self, dim=None, unbiased=True, keepdim=False):
        """See :func:`~torch.std`."""
        if not self:
            if keepdim:
                return torch.Tensor([float("nan")], device=self.device).expand(
                    [1 for i in range(self.dim())]
                )
            else:
                return torch.Tensor([float("nan")], device=self.device)
        if isinstance(dim, Iterable):  # Fix broken iterables.
            if not dim:
                dim = None
            elif len(dim) == 1:
                dim = dim[0]

        if dim is None:
            mean = self.mean()
            result = (
                (self - mean).pow(2).sum()
                / (self.numel() - 1 if unbiased else self.numel())
            ).sqrt()
            if keepdim:
                return result.expand([1 for i in range(self.dim())])
            else:
                return result
        # Check for dim 0 first.
        if isinstance(dim, Iterable):
            if 0 in dim or -self.dim() in dim:
                # The case where the contained item is TensorList must
                # have the same shape and so it returns a full tensor.
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().std(
                    dim=dim, unbiased=unbiased, keepdim=keepdim
                )
        elif dim == 0 or dim == -self.dim():
            if isinstance(self[0], TensorList):
                mean = self.mean(dim=0)
                result = (
                    sum((item - mean).pow(2) for item in self)
                    / (len(self) - 1 if unbiased else len(self))
                ).sqrt()
                if keepdim:
                    return self.__class__([result])
                else:
                    return result
            else:
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().std(
                    dim=0, unbiased=unbiased, keepdim=keepdim
                )

        # No dim 0 present.
        if isinstance(dim, Iterable):
            dim = [(d - 1 if d > 0 else d) for d in dim]
        else:
            dim = dim - 1 if dim > 0 else dim
        return self.__class__(
            [
                item.std(dim=dim, unbiased=unbiased, keepdim=keepdim)
                for item in self
            ]
        )

    def sum(self, dim=None, keepdim=False, dtype=None):
        """See :func:`~torch.sum`."""
        if not self:
            if keepdim:
                return (
                    torch.Tensor([0])
                    .to(dtype=dtype)
                    .expand([1 for i in range(self.dim())])
                )
            else:
                return torch.Tensor([0]).to(dtype=dtype)
        if isinstance(dim, Iterable):  # Fix broken iterables.
            if not dim:
                dim = None
            elif len(dim) == 1:
                dim = dim[0]

        if dim is None:
            result = sum(item.sum(dtype=dtype) for item in self)
            if keepdim:
                return result.expand([1 for i in range(self.dim())])
            else:
                return result
        # Check for dim 0 first.
        if isinstance(dim, Iterable):
            if 0 in dim or -self.dim() in dim:
                # The case where the contained item is TensorList must
                # have the same shape and so it returns a full tensor.
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().sum(
                    dim=dim, keepdim=keepdim, dtype=dtype
                )
        elif dim == 0 or dim == -self.dim():
            if isinstance(self[0], TensorList):
                result = self.__class__(
                    [
                        sum(item[l].to(dtype=dtype) for item in self)
                        for l in range(len(self[0]))
                    ]
                )
                if keepdim:
                    return self.__class__([result])
                else:
                    return result
            else:
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().sum(
                    dim=0, keepdim=keepdim, dtype=dtype
                )

        # No dim 0 present.
        if isinstance(dim, Iterable):
            dim = [(d - 1 if d > 0 else d) for d in dim]
        else:
            dim = dim - 1 if dim > 0 else dim
        return self.__class__(
            [item.sum(dim=dim, keepdim=keepdim, dtype=dtype) for item in self]
        )

    def t(self):
        """See :func:`~torch.t`."""
        # Transpose only works if this is actually a 2D tensor.
        # This may fail if the Tensors are not of the same shape.
        return self.to_tensor().permute(*dims)

    def to(self, *args, **kwargs):
        """See :func:`~torch.Tensor.to`."""
        result = [item.to(*args, **kwargs) for item in self]
        if "copy" in kwargs and kwargs["copy"]:
            # Force creation of new TensorList.
            return self.__class__(result)
        else:
            self.data = result
            return self

    def tolist(self):
        """See :func:`~torch.Tensor.tolist`."""
        return [item.tolist() for item in self]

    def transpose(self, dim0, dim1):
        """See :func:`~torch.transpose`."""
        if dim0 < 0:
            dim0 += self.dim()
        if dim1 < 0:
            dim1 += self.dim()
        # Sort dims (does not change anything).
        if dim0 > dim1:
            dim0, dim1 = dim1, dim0
        if dim0 == 0:
            if dim1 == 0:
                return self  # No-op.
            elif dim1 == 1 and self and isinstance(self[0], TensorList):
                # TensorList permutation only implemented for first two levels.
                return self.__class__(
                    [
                        self.__class__([self[i][j] for i in range(len(self))])
                        for j in range(len(self[0]))
                    ]
                )
            else:  # Data must be contiguous.
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().transpose(dim0, dim1)
        else:
            return self.__class__(
                [item.transpose(dim0 - 1, dim1 - 1) for item in self]
            )

    def transpose_(self, dim0, dim1):
        """In-place version of :func:`~TensorList.transpose`."""
        if dim0 < 0:
            dim0 += self.dim()
        if dim1 < 0:
            dim1 += self.dim()
        # Sort dims (does not change anything).
        if dim0 > dim1:
            dim0, dim1 = dim1, dim0
        if dim0 == 0:
            if dim1 == 0:
                return self  # No-op.
            elif dim1 == 1 and self and isinstance(self[0], TensorList):
                # TensorList permutation only implemented for first two levels.
                self.data = [
                    self.__class__([self[i][j] for i in range(len(self))])
                    for j in range(len(self[0]))
                ]
            else:  # Data must be contiguous.
                raise RuntimeError(
                    "Cannot use in-place transpose as that would change type"
                )
        else:
            self.data = [item.transpose(dim0 - 1, dim1 - 1) for item in self]
        return self

    def unsqueeze(self, dim):
        """See :func:`~torch.unsqueeze`."""
        if dim < 0:  # Per torch docs.
            dim += self.dim() + 1
        if dim == 0:
            return self.__class__([self])
        else:
            return self.__class__([item.unsqueeze(dim - 1) for item in self])

    def unsqueeze_(self, dim):
        """In-place version of :func:`~TensorList.unsqueeze`."""
        if dim < 0:  # Per torch docs.
            dim += self.dim() + 1
        if dim == 0:
            inner = self.__class__()
            inner.data = self.data
            self.data = [inner]
        else:
            self.data = [item.unsqueeze(dim - 1) for item in self]
        return self

    def var(self, dim=None, unbiased=True, keepdim=False):
        """See :func:`~torch.var`."""
        if not self:
            if keepdim:
                return torch.Tensor([float("nan")], device=self.device).expand(
                    [1 for i in range(self.dim())]
                )
            else:
                return torch.Tensor([float("nan")], device=self.device)
        if isinstance(dim, Iterable):  # Fix broken iterables.
            if not dim:
                dim = None
            elif len(dim) == 1:
                dim = dim[0]

        if dim is None:
            mean = self.mean()
            result = (self - mean).pow(2).sum() / (
                self.numel() - 1 if unbiased else self.numel()
            )
            if keepdim:
                return result.expand([1 for i in range(self.dim())])
            else:
                return result
        # Check for dim 0 first.
        if isinstance(dim, Iterable):
            if 0 in dim or -self.dim() in dim:
                # The case where the contained item is TensorList must
                # have the same shape and so it returns a full tensor.
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().var(
                    dim=dim, unbiased=unbiased, keepdim=keepdim
                )
        elif dim == 0 or dim == -self.dim():
            if isinstance(self[0], TensorList):
                mean = self.mean(dim=0)
                result = sum((item - mean).pow(2) for item in self) / (
                    len(self) - 1 if unbiased else len(self)
                )
                if keepdim:
                    return self.__class__([result])
                else:
                    return result
            else:
                # This may fail if the Tensors are not of the same shape.
                return self.to_tensor().var(
                    dim=0, unbiased=unbiased, keepdim=keepdim
                )

        # No dim 0 present.
        if isinstance(dim, Iterable):
            dim = [(d - 1 if d > 0 else d) for d in dim]
        else:
            dim = dim - 1 if dim > 0 else dim
        return self.__class__(
            [
                item.var(dim=dim, unbiased=unbiased, keepdim=keepdim)
                for item in self
            ]
        )

    def view(self, *shape):
        """Unavailable: Use :func:`~TensorList.reshape` instead."""
        raise NotImplementedError(
            "TensorList.view requires contiguous storage, use reshape instead"
        )

    @classmethod
    def cat(cls, tensors, dim=0, out=None):
        """See :func:`~torch.cat`."""
        if out is not None:
            raise ValueError("out is not supported in TensorList cat.")
        if not tensors:
            raise ValueError("cat called with an empty list of tensors.")
        if not isinstance(tensors[0], TensorList):
            for i, t in enumerate(tensors):
                if not isinstance(t, torch.Tensor):
                    raise TypeError(
                        "expected Tensor as element {} in cat".format(i)
                    )
            # Fall back to torch.cat if they are of the same shape:
            if all(t.shape[:dim] == tensors[0].shape[:dim] for t in tensors):
                return torch.cat(tensors, dim=dim)
            # Else must insert a TensorList for all dimensions up to dim.
            # This is handled by the below code for both Tensors & TensorLists.

        if dim == 0 or dim == -tensors[0].dim():
            tensors = [item for t in tensors for item in t]
        else:
            dim = dim - 1 if dim > 0 else dim
            # This will fail if tensors do not have matching shapes.
            tensors = [
                cls.cat([t[i] for t in tensors], dim=dim)
                for i in range(len(tensors[0]))
            ]
        # Stack resulting tensors along dimension 0.
        if (
            tensors
            and (not isinstance(tensors[0], TensorList))
            and all(t.shape == tensors[0].shape for t in tensors)
        ):
            return torch.stack(tensors, dim=0)
        else:
            # Have to support tensorlists or unequal tensors as children
            return cls(tensors)
