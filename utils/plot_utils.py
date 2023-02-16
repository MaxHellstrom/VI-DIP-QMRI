from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class GridPlot:
    def __init__(self, nrows=3, ncols=3, size=None, lims=[None, None], dpi=None):

        self.nrows, self.ncols = nrows, ncols
        self.size, self.lims = size, lims
        self.lims = lims
        self.cmap = plt.cm.gray
        self.latex = False

        self.fig, self.axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=size, dpi=dpi
        )

        self.lines = GridPlot.Lines(self)
        self.ticks = GridPlot.Ticks(self)
        self.cmaps = GridPlot.Cmaps()
        self.cbar = GridPlot.Cbar(self)
        self.text = GridPlot.Text(self)

    def set_spacing(self, w=None, h=None, l=None, r=None, t=None, b=None):
        plt.subplots_adjust(wspace=w, hspace=h, left=l, right=r, top=t, bottom=b)

    def set_size(self, x, y):
        self.fig.set_size_inches(x, y)

    def set_axs_shapes(self, shape=(100, 100)):

        for row, col in product(range(self.nrows), range(self.ncols)):
            _set_ax_shape(ax=self.axs[row, col], shape=shape)

    def export(self, file_name, bbox_inches=None, dpi=None):

        self.fig.savefig(fname=file_name, bbox_inches=bbox_inches, dpi=dpi)

    def add_subplot(
        self,
        row,
        col,
        mat,
        lims=[None, None],
        cmap=None,
        xlabel="",
        ylabel="",
        title="",
        cbar=True,
        cbar_ticks=None,
        cbar_label="",
        cbar_binary=False,
        binary_ticklabels = ['excl.', 'incl.']
    ):

        cmap = self.cmap if cmap is None else cmap

        if lims == [None, None]:
            lims = self.lims

        if self.latex:
            xlabel, ylabel = _tex_str(xlabel), _tex_str(ylabel)
            title, cbar_label = _tex_str(title), _tex_str(cbar_label)
            binary_ticklabels[0] = _tex_str(binary_ticklabels[0])
            binary_ticklabels[1] = _tex_str(binary_ticklabels[1])

        self.axs[row, col].im = _create_subplot(
            ax=self.axs[row, col],
            mat=mat,
            lims=lims,
            cmap=cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        if cbar:
            self.axs[row, col].cbar = _create_colorbar(
                fig=self.fig,
                im=self.axs[row, col].im,
                ax=self.axs[row, col],
                ticks=cbar_ticks,
                label=cbar_label,
                borderpad=-1,
                binary=cbar_binary,
                binary_ticklabels=binary_ticklabels
            )

    class Text:
        def __init__(self, obj):
            self.obj = obj

        def xlabel_to_row(self, row, labels):
            for row, col in product([row], range(self.obj.ncols)):
                
                if labels[col] is None:
                    continue
                
                if len(self.obj.axs.shape) == 1:
                    ax = self.obj.axs[col]
                
                else:
                    ax = self.obj.axs[row, col]

                _set_ax_text(
                    ax=ax,
                    xlabel=_tex_str(labels[col]) if self.obj.latex else labels[col],
                )

        def ylabel_to_col(self, col, labels, disable_tex=False):
            
            tex = self.obj.latex if not disable_tex else False
            for row, col in product(range(self.obj.nrows), [col]):

                if labels[row] is None:
                    continue

                if len(self.obj.axs.shape) == 1:
                    ax = self.obj.axs[row]

                else:
                    ax = self.obj.axs[row, col]

                _set_ax_text(
                    ax=ax,
                    ylabel=_tex_str(labels[row]) if tex else labels[row],
                )

        def ylabel_to_ax(self, row, col, label):
            _set_ax_text(
                ax=self.obj.axs[row, col],
                ylabel=_tex_str(label) if self.obj.latex else label,
            )

        def titles_to_row(self, row, titles):
            for row, col in product([row], range(self.obj.ncols)):
                if titles[col] is None:
                    continue
                
                if len(self.obj.axs.shape) == 1:
                    ax = self.obj.axs[col]

                else:
                    ax = self.obj.axs[row, col]
                
                _set_ax_text(
                    ax=ax,
                    title=_tex_str(titles[col]) if self.obj.latex else titles[col],
                )

        def plotlabel_to_row(
            self,
            row,
            labels,
            px=0.05,
            py=0.94,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", ec="black", fc="white"),
            latex_bold=True,
        ):

            for row, col in product([row], range(self.obj.ncols)):
                if labels[col] is None:
                    continue

                _add_label_to_ax(
                    ax=self.obj.axs[row, col],
                    px=px,
                    py=py,
                    ha=ha,
                    va=va,
                    fontsize=fontsize,
                    bbox=bbox,
                    label=_tex_str(labels[col], bold=latex_bold)
                    if self.obj.latex
                    else labels[col],
                )

        def plotlabel_to_ax(
            self,
            row,
            col,
            label,
            px=0.05,
            py=0.94,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", ec="black", fc="white"),
            latex_bold=True,
        ):

            _add_label_to_ax(
                ax=self.obj.axs[row, col],
                px=px,
                py=py,
                ha=ha,
                va=va,
                fontsize=fontsize,
                bbox=bbox,
                label=_tex_str(label, bold=latex_bold) if self.obj.latex else label,
            )

    class Cmaps:
        def __init__(self):
            self.mag = plt.cm.gray
            self.bias = plt.cm.RdBu.reversed()
            self.std = plt.cm.magma
            self.binary = matplotlib.colors.ListedColormap(["black", "white"])

    class Cbar:
        def __init__(self, obj):
            self.obj = obj

        def remove(self, row, col):
            self.obj.axs[row, col].cbar.remove()

        def remove_all(self):
            for row, col in product(range(self.obj.nrows), range(self.obj.ncols)):
                if hasattr(self.obj.axs[row, col], "cbar"):
                    self.obj.axs[row, col].cbar.remove()

        def remove_rows(self, rows):
            rows = [rows] if isinstance(rows, int) else rows
            for row, col in product(rows, range(self.obj.ncols)):
                if hasattr(self.obj.axs[row, col], "cbar"):
                    self.obj.axs[row, col].cbar.remove()

        def remove_cols(self, cols):
            cols = [cols] if isinstance(cols, int) else cols
            for row, col in product(range(self.obj.nrows), cols):
                if hasattr(self.obj.axs[row, col], "cbar"):
                    self.obj.axs[row, col].cbar.remove()

        def row_common(
            self,
            row,
            width="10%",
            height="100%",
            loc="center right",
            bbox_to_anchor=(0, 0, 1, 1),
            borderpad=-1,
            orientation="vertical",
            ticks=None,
            label=None,
            ticklabels=None,
            title=None,
            binary=False,
            binary_ticklabels = ['excl.', 'incl.']
        ):

            for row, col in product([row], range(self.obj.ncols)):
                if hasattr(self.obj.axs[row, col], "cbar"):
                    self.obj.axs[row, col].cbar.remove()
            
            if self.obj.latex:
                if binary:
                    binary_ticklabels[0] = _tex_str(binary_ticklabels[0])
                    binary_ticklabels[1] = _tex_str(binary_ticklabels[1])

                if label is not None:
                    label = _tex_str(label)

            _create_colorbar(
                fig=self.obj.fig,
                im=self.obj.axs[row, self.obj.ncols - 1].im,
                ax=self.obj.axs[row, self.obj.ncols - 1],
                width=width,
                height=height,
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                borderpad=borderpad,
                orientation=orientation,
                ticks=ticks,
                label=label,
                ticklabels=ticklabels,
                title=title,
                binary=binary,
                binary_ticklabels=binary_ticklabels

            )

        def col_common(
            self,
            col,
            width="100%",
            height="10%",
            loc="lower center",
            bbox_to_anchor=(0, 0, 1, 1),
            borderpad=-1,
            orientation="horizontal",
            ticks=None,
            label=None,
            ticklabels=None,
            title=None,
            binary=False,
            binary_ticklabels = ['excl.', 'incl.']
        ):

            for row, col in product(range(self.obj.nrows), [col]):
                if hasattr(self.obj.axs[row, col], "cbar"):
                    self.obj.axs[row, col].cbar.remove()

            if self.obj.latex:
                if binary:
                    binary_ticklabels[0] = _tex_str(binary_ticklabels[0])
                    binary_ticklabels[1] = _tex_str(binary_ticklabels[1])

                if label is not None:
                    label = _tex_str(label)

            _create_colorbar(
                fig=self.obj.fig,
                im=self.obj.axs[self.obj.nrows - 1, col].im,
                ax=self.obj.axs[self.obj.nrows - 1, col],
                width=width,
                height=height,
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                borderpad=borderpad,
                orientation=orientation,
                ticks=ticks,
                label=label,
                ticklabels=ticklabels,
                title=title,
                binary=binary,
                binary_ticklabels=binary_ticklabels
            )

    class Ticks:
        def __init__(self, obj):
            self.obj = obj

        def remove(self, row, col):
            _remove_ax_ticks(ax=self.obj.axs[row, col])

        def remove_all(self, exceptions=[]):
            for row, col in product(range(self.obj.nrows), range(self.obj.ncols)):
                if (row, col) not in exceptions:
                    _remove_ax_ticks(ax=self.obj.axs[row, col])

        def remove_rows(self, rows):
            rows = [rows] if isinstance(rows, int) else rows
            for row, col in product(rows, range(self.obj.ncols)):
                _remove_ax_ticks(ax=self.obj.axs[row, col])

        def remove_cols(self, cols):
            cols = [cols] if isinstance(cols, int) else cols
            for row, col in product(range(self.obj.nrows), cols):
                _remove_ax_ticks(ax=self.obj.axs[row, col])

    class Lines:
        def __init__(self, obj):
            self.obj = obj

        def remove(self, row, col):
            _remove_ax_lines(ax=self.obj.axs[row, col])

        def remove_all(self):
            for row, col in product(range(self.obj.nrows), range(self.obj.ncols)):
                _remove_ax_lines(ax=self.obj.axs[row, col])

        def remove_rows(self, rows):
            rows = [rows] if isinstance(rows, int) else rows
            for row, col in product(rows, range(self.obj.ncols)):
                _remove_ax_lines(ax=self.obj.axs[row, col])

        def remove_cols(self, cols):
            cols = [cols] if isinstance(cols, int) else cols
            for row, col in product(range(self.obj.nrows), cols):
                _remove_ax_lines(ax=self.obj.axs[row, col])

        def set_color(self, row, col, color):
            _set_ax_line_color(ax=self.obj.axs[row, col], color=color)

        def remove_inner(self):
            for row, col in product(range(self.obj.nrows), range(self.obj.ncols)):
                if col > 0:
                    _remove_ax_lines(ax=self.obj.axs[row, col], dirs=["left"])
                if col < self.obj.ncols - 1:
                    _remove_ax_lines(ax=self.obj.axs[row, col], dirs=["right"])
                if row > 0:
                    _remove_ax_lines(ax=self.obj.axs[row, col], dirs=["top"])
                if row < self.obj.nrows - 1:
                    _remove_ax_lines(ax=self.obj.axs[row, col], dirs=["bottom"])


def _create_colorbar(
    fig,
    im,
    ax,
    width="10%",
    height="100%",
    loc="center right",
    bbox_to_anchor=(0.05, 0, 1, 1),
    borderpad=0,
    orientation="vertical",
    ticks=None,
    label=None,
    binary=False,
    ticklabels=None,
    title=None,
    binary_ticklabels=["excl", "incl"],
):

    axins = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        borderpad=borderpad,
    )

    cbar = fig.colorbar(
        im,
        cax=axins,
        ticks=ticks,
        orientation=orientation,
        label=label,
    )

    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels=ticklabels)

    if title is not None:
        cbar.ax.set_title(title)

    if binary:
        cbar.set_ticks(ticks=[1 / 4, 3 / 4])
        cbar.set_ticklabels(ticklabels=binary_ticklabels)

    return cbar


def _remove_ax_lines(ax, dirs=["left", "right", "top", "bottom"]):
    for d in dirs:
        ax.spines[d].set_visible(False)


def _remove_ax_ticks(ax):
    ax.set(xticks=[], yticks=[])


def _set_ax_text(ax, xlabel=None, ylabel=None, title=None):

    ax.set_xlabel(xlabel) if xlabel is not None else None
    ax.set_ylabel(ylabel) if ylabel is not None else None
    ax.set_title(title) if title is not None else None


def _add_label_to_ax(
    ax,
    label,
    px=0.05,
    py=0.94,
    ha="left",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round", ec="black", fc="white"),
):

    ax.text(
        px,
        py,
        s=label,
        ha=ha,
        va=va,
        fontsize=fontsize,
        transform=ax.transAxes,
        bbox=bbox,
    )


def _set_ax_line_color(ax, color, dirs=["left", "right", "top", "bottom"]):
    for d in dirs:
        ax.spines[d].set_color(color)


def _create_subplot(ax, mat, lims, cmap, xlabel, ylabel, title):

    im = ax.imshow(mat, vmin=lims[0], vmax=lims[1], cmap=cmap, interpolation=None)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return im


def _set_ax_shape(ax, shape, white=True):

    return ax.imshow(
        np.ones(shape), cmap=plt.cm.gray if white else plt.cm.binary, vmin=0, vmax=1
    )


def _plotdpi(dpi=124):
    return int(dpi)


def _create_bias_image(mean_image, ref, mask=None) -> np.array:
    """Helper function to create bias maps"""

    error_img = np.zeros(ref.shape)

    if mask is None:
        mask = ref != 0
    error_img[mask] = 100 * (mean_image[mask] - ref[mask]) / ref[mask]

    return error_img


def _create_percentage_diff_image(image, ref, mask):

    res = np.zeros(mask.shape)
    res[mask] = image[mask] * (100 / ref[mask])

    return res


def _draw_region_box(c, sx, sy):

    cx, cy = c[0], c[1]

    trace_x = np.array(
        [cx - 0.5 * sx, cx - 0.5 * sx, cx + 0.5 * sx, cx + 0.5 * sx, cx - 0.5 * sx]
    )

    trace_y = np.array(
        [cy - 0.5 * sy, cy + 0.5 * sy, cy + 0.5 * sy, cy - 0.5 * sy, cy - 0.5 * sy]
    )

    return trace_x.astype(np.int), trace_y.astype(np.int)


def _select_matrix_from_box(image, trace_x, trace_y, stack=False):

    xmin, xmax, ymin, ymax = trace_x.min(), trace_x.max(), trace_y.min(), trace_y.max()

    if stack:
        return image[:, ymin:ymax, xmin:xmax]
    return image[ymin:ymax, xmin:xmax]


def _plotsize(width=8.25, height=11.75, indent_width=0, indent_height=0):
    # defaults are a4 paper dimensions (inches)
    return (width - indent_width, height - indent_height)


def _tex_str(string, bold=False):
    # note: replace \ with \\ in argument string
    if bold:
        return "\\textnormal{" + "\\textbf{" + string + "}}"
    return "\\textnormal{" + string + "}"


def _cmap_binary(c0="black", c1="white"):
    "function to set binary color map (e.g. for image masks)"
    return matplotlib.colors.ListedColormap([c0, c1])


def _padding(array, xx, yy, constant_values=0):

    h, w = array.shape[0], array.shape[1]
    a = (xx - h) // 2
    aa = xx - a - h
    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(
        array,
        pad_width=((a, aa), (b, bb)),
        mode="constant",
        constant_values=constant_values,
    )


def plot_estimation_progress(
    P_ref,
    P_NLLS,
    P_est,
    P_epi_std,
    y,
    y_denoised,
    mask,
    iter,
    loss_list,
    save_path,
    param_names=["P0", "P1"],
    param_units=["P0 units", "P1 units"],
    image_type="pdf",
    suptitle=None,
    noise_std=None,
    LR=None,
):

    plot = GridPlot(nrows=6, ncols=5)
    plot.set_size(x=20, y=20)
    plot.fig.suptitle(suptitle)
    exceptions = [(0, 4)]
    plot.ticks.remove_all(exceptions=exceptions)

    fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(20, 20))  #

    fig.suptitle(suptitle)

    error_prct = 50
    lim_error = [-error_prct, +error_prct]
    empty_lim = [None, None]
    lim_P = []

    for param_index in [0, 1]:
        lim_P.append(
            [0, max([P_ref[param_index, :, :].max(), P_NLLS[param_index, :, :].max()])]
        )

    for col in range(y.shape[0]):
        # plot input signal
        plot.add_subplot(
            row=3,
            col=col,
            mat=mask * y[col, :, :],
            lims=empty_lim,
            cmap=plot.cmaps.mag,
            title=f"s in: {col+1}, std: {round(noise_std,  5)}",
        )

        # plot denoised signal
        refmat = mask * y[col, :, :]
        plot.add_subplot(
            row=4,
            col=col,
            mat=mask * y_denoised[col, :, :],
            lims=[refmat.min(), refmat.max()],
            cmap=plot.cmaps.mag,
            title=f"s out: {col+1}",
        )

        # plot signal diff from input
        plot.add_subplot(
            row=5,
            col=col,
            mat=mask
            * _create_bias_image(
                mean_image=y_denoised[col, :, :], ref=y[col, :, :], mask=mask
            ),
            lims=lim_error,
            cmap=plot.cmaps.bias,
            title=f"s diff from input: {col+1}",
        )

    for param_index in range(2):

        # plot P est
        plot.add_subplot(
            row=0,
            col=param_index,
            mat=mask * P_est[param_index, :, :],
            lims=empty_lim,
            cmap=plot.cmaps.mag,
            title=f"{param_names[param_index]} est [{param_units[param_index]}]",
        )

        # plot P epi std
        plot.add_subplot(
            row=0,
            col=2 + param_index,
            mat=mask * P_epi_std[param_index, :, :],
            lims=empty_lim,
            cmap=plot.cmaps.std,
            title=f"{param_names[param_index]} epi std [{param_units[param_index]}]",
        )

        # plot P NLLS
        plot.add_subplot(
            row=1,
            col=param_index,
            mat=mask * P_NLLS[param_index, :, :],
            lims=lim_P[param_index],
            cmap=plot.cmaps.mag,
            title=f"{param_names[param_index]} NLLS [{param_units[param_index]}]",
        )

        # plot P ref
        plot.add_subplot(
            row=1,
            col=2 + param_index,
            mat=mask * P_ref[param_index, :, :],
            lims=lim_P[param_index],
            cmap=plot.cmaps.mag,
            title=f"{param_names[param_index]} ref [{param_units[param_index]}]",
        )

        # plot P est diff from NLLS
        plot.add_subplot(
            row=2,
            col=param_index,
            mat=_create_bias_image(
                mean_image=P_est[param_index, :, :],
                ref=P_NLLS[param_index, :, :],
                mask=mask,
            ),
            lims=lim_error,
            cmap=plot.cmaps.bias,
            title=f"{param_names[param_index]} est diff from NLLS [%]",
        )

        # plot P est diff from ref
        plot.add_subplot(
            row=2,
            col=param_index + 2,
            mat=_create_bias_image(
                mean_image=P_est[param_index, :, :],
                ref=P_ref[param_index, :, :],
                mask=mask,
            ),
            lims=lim_error,
            cmap=plot.cmaps.bias,
            title=f"{param_names[param_index]} est diff from ref [%]",
        )

    # plot MSE error
    plot.axs[0, 4].plot(np.asarray(loss_list), "k")
    plot.axs[0, 4].set(title="training loss loss")
    if iter > 0:
        plot.axs[0, 4].set_yscale("log")
    plot.axs[0, 4].yaxis.tick_right()
    plot.axs[0, 4].set(title=f"LR: {LR}")

    # plot mask
    plot.add_subplot(
        row=1,
        col=4,
        mat=mask,
        cmap=plot.cmaps.binary,
        cbar_binary=True,
        title=f"mask with shape {mask.shape}",
    )

    plt.subplots_adjust(wspace=0.5, left=0.01, right=0.95, top=0.95, bottom=0.025)
    plot.fig.savefig(f"{save_path}progress_iteration_{iter}.{image_type}")
    plt.close()


def _make_bg_under(mat, mask, lim=0):
    mat[~mask] = lim-0.01
    return mat


def _pad_along_axis(
    array: np.ndarray,
    target_length: int,
    axis: int = 0,
    rotate_180=False,
    constant_value=0,
):

    if rotate_180:
        array = np.rot90(np.rot90(array))

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    if rotate_180:
        return np.rot90(
            np.rot90(
                np.pad(
                    array,
                    pad_width=npad,
                    mode="constant",
                    constant_values=constant_value,
                )
            )
        )
    return np.pad(
        array, pad_width=npad, mode="constant", constant_values=constant_value
    )
