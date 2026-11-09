from dataclasses import dataclass


TOKEN_LAYOUT_64_VERSION = "geometrize_64_v1"
TOKEN_LAYOUT_192_VERSION = "geometrize_192_v1"
LEGACY_TOKEN_LAYOUT_VERSION = "geometrize_256_v1"
DATA_TOKEN_LAYOUT_VERSION = "geometrize_512_three_stage_interrupt_v1"
TOKEN_LAYOUT_VERSION = "geometrize_512_v2_primitive_embedding_end_head_interrupt_bos"
# 2026-11 bezier 扩展：矩形/椭圆四边各加一个贝塞尔控制点。
TOKEN_LAYOUT_VERSION_BEZIER = "geometrize_512_v3_bezier_edges"
PREVIOUS_TOKEN_LAYOUT_VERSION = "geometrize_256_v4_primitive_embedding_marker_end"

XY_BINS_PER_PIXEL = 2
ANGLE_BINS_PER_DEGREE = 3
SIZE_BINS_PER_PIXEL = 4


@dataclass(frozen=True)
class GeometrizeTokenLayout:
    x_bins: int
    y_bins: int
    angle_bins: int
    width_bins: int
    height_bins: int
    shape_bins: int = 256
    color_bins: int = 128
    edge_bins: int = 0
    edge_count: int = 4
    special_bins: int = 3
    empty_fields: tuple = ()
    marker_end_bins: int = 0
    interrupt_end_bins: int = 0
    bos_bins: int = 0
    version: str = "unknown"

    @property
    def x_offset(self):
        return 0

    @property
    def y_offset(self):
        return self.x_offset + self.x_bins

    @property
    def angle_offset(self):
        return self.y_offset + self.y_bins

    @property
    def width_offset(self):
        return self.angle_offset + self.angle_bins

    @property
    def height_offset(self):
        return self.width_offset + self.width_bins

    @property
    def shape_offset(self):
        return self.height_offset + self.height_bins

    @property
    def red_offset(self):
        return self.shape_offset + self.shape_bins

    @property
    def green_offset(self):
        return self.red_offset + self.color_bins

    @property
    def blue_offset(self):
        return self.green_offset + self.color_bins

    @property
    def edge_offset(self):
        """贝塞尔边控制点的起始 token 偏移；4 条边按 edge0..edge3 顺序排列。"""
        return self.blue_offset + self.color_bins

    @property
    def edge_offsets(self):
        return tuple(
            self.edge_offset + index * self.edge_bins
            for index in range(self.edge_count)
        )

    @property
    def special_offset(self):
        return self.edge_offset + self.edge_bins * self.edge_count

    @property
    def vocab_size(self):
        return self.bos_offset + self.bos_bins

    @property
    def attribute_vocab_size(self):
        """Output-capable vocabulary; control and partial tokens are input-only."""
        return self.special_offset

    @property
    def pad_token(self):
        return self.special_offset

    @property
    def region_marker_token(self):
        """RGB 字段专用的区域条件 token（追加在历史词表末尾）。"""
        return self.special_offset + self.special_bins - 1

    @property
    def marker_begin_token(self):
        """Backward-compatible name for the RGB marker-begin token."""
        return self.region_marker_token

    @property
    def empty_offset(self):
        """Start of input-only field-specific empty tokens."""
        return self.special_offset + self.special_bins

    @property
    def empty_token_ids(self):
        return tuple(self.empty_offset + index for index in range(len(self.empty_fields)))

    @property
    def marker_end_offset(self):
        """Input-only true-end carrier appended after field-specific empties."""
        return self.empty_offset + len(self.empty_fields)

    @property
    def marker_end_token(self):
        if self.marker_end_bins != 1:
            raise ValueError("当前词表布局未定义独立 marker_end token。")
        return self.marker_end_offset

    @property
    def true_end_token(self):
        """Preferred name for the learned input carrier of a real block end."""
        return self.marker_end_token

    @property
    def interrupt_end_offset(self):
        return self.marker_end_offset + self.marker_end_bins

    @property
    def interrupt_end_token(self):
        if self.interrupt_end_bins != 1:
            raise ValueError("当前词表布局未定义独立 interrupt_end token。")
        return self.interrupt_end_offset

    @property
    def bos_offset(self):
        return self.interrupt_end_offset + self.interrupt_end_bins

    @property
    def bos_token(self):
        if self.bos_bins != 1:
            raise ValueError("当前词表布局未定义图元 BOS。")
        return self.bos_offset

    def empty_token(self, field_name):
        if field_name == "x":
            raise ValueError("x 由前一完整图元预测，不存在 x 空 token。")
        try:
            return self.empty_offset + self.empty_fields.index(field_name)
        except ValueError as error:
            raise KeyError(f"字段 {field_name!r} 没有专属空 token。") from error

    def field_slice(self, field_name):
        if field_name.startswith("edge") and field_name[4:].isdigit():
            index = int(field_name[4:])
            if index >= self.edge_count:
                raise KeyError(
                    f"边控制点 {field_name!r} 超出 edge_count={self.edge_count}"
                )
            start = self.edge_offset + index * self.edge_bins
            return slice(start, start + self.edge_bins)
        offset = getattr(self, f"{field_name}_offset")
        if field_name in {"red", "green", "blue"}:
            bins = self.color_bins
        else:
            bins = getattr(self, f"{field_name}_bins")
        return slice(offset, offset + bins)

    def to_metadata(self):
        return {
            "version": self.version,
            "vocab_size": self.vocab_size,
            "bins": {
                "x": self.x_bins,
                "y": self.y_bins,
                "angle": self.angle_bins,
                "width": self.width_bins,
                "height": self.height_bins,
                "shape": self.shape_bins,
                "red": self.color_bins,
                "green": self.color_bins,
                "blue": self.color_bins,
                "edge": self.edge_bins,
                "special": self.special_bins,
                "empty": len(self.empty_fields),
                "marker_end": self.marker_end_bins,
                "interrupt_end": self.interrupt_end_bins,
                "bos": self.bos_bins,
            },
            "offsets": {
                "x": self.x_offset,
                "y": self.y_offset,
                "angle": self.angle_offset,
                "width": self.width_offset,
                "height": self.height_offset,
                "shape": self.shape_offset,
                "red": self.red_offset,
                "green": self.green_offset,
                "blue": self.blue_offset,
                "edge": self.edge_offset,
                "special": self.special_offset,
                "empty": self.empty_offset,
                "marker_end": self.marker_end_offset,
                "interrupt_end": self.interrupt_end_offset,
                "bos": self.bos_offset,
            },
            "empty_fields": list(self.empty_fields),
            "empty_token_ids": list(self.empty_token_ids),
            "marker_end_token": (
                self.marker_end_token if self.marker_end_bins else None
            ),
            "true_end_token": (
                self.true_end_token if self.marker_end_bins else None
            ),
            "interrupt_end_token": (
                self.interrupt_end_token if self.interrupt_end_bins else None
            ),
            "bos_token": self.bos_token if self.bos_bins else None,
            "attribute_vocab_size": self.attribute_vocab_size,
        }


# 64x64 checkpoint 中实际使用的历史词表布局。
TOKEN_LAYOUT_64 = GeometrizeTokenLayout(
    x_bins=129,
    y_bins=129,
    angle_bins=181,
    width_bins=193,
    height_bins=193,
    version=TOKEN_LAYOUT_64_VERSION,
)

# 192x192 checkpoint 的源词表布局。宽高仍只表示 [0, 128)。
TOKEN_LAYOUT_192 = GeometrizeTokenLayout(
    x_bins=384,
    y_bins=384,
    angle_bins=270,
    width_bins=512,
    height_bins=512,
    version=TOKEN_LAYOUT_192_VERSION,
)

# epoch 157 的源布局，只用于严格的语义权重迁移。
TOKEN_LAYOUT_256 = GeometrizeTokenLayout(
    x_bins=512,
    y_bins=512,
    angle_bins=270,
    width_bins=512,
    height_bins=512,
    special_bins=4,
    empty_fields=(
        "y", "angle", "width", "height", "shape", "red", "green", "blue"
    ),
    marker_end_bins=1,
    version=PREVIOUS_TOKEN_LAYOUT_VERSION,
)

# 512x512 布局按右开区间编码：坐标 [0,512)，完整宽高 [0,512)。
TOKEN_LAYOUT = GeometrizeTokenLayout(
    x_bins=1024,
    y_bins=1024,
    angle_bins=270,
    width_bins=2048,
    height_bins=2048,
    edge_bins=256,
    edge_count=4,
    special_bins=4,
    empty_fields=(
        "y", "angle", "width", "height", "shape", "red", "green", "blue",
        "edge0", "edge1", "edge2", "edge3",
    ),
    marker_end_bins=1,
    interrupt_end_bins=1,
    bos_bins=1,
    version=TOKEN_LAYOUT_VERSION_BEZIER,
)

# 兼容历史迁移工具的旧名称。
OLD_TOKEN_LAYOUT = TOKEN_LAYOUT_64


if TOKEN_LAYOUT_64.vocab_size != 1468:
    raise RuntimeError("64x64 历史词表布局计算错误")
if TOKEN_LAYOUT_192.vocab_size != 2705:
    raise RuntimeError("192x192 词表布局计算错误")
if TOKEN_LAYOUT_256.vocab_size != 2971:
    raise RuntimeError("256x256 源词表布局计算错误")
# bezier 扩展后的 512 词表：属性 7054 + 边控制点 4x256 = 8078，
# special 4 + empty 12 + marker_end/interrupt_end/BOS 3 = 8097。
if TOKEN_LAYOUT.attribute_vocab_size != 8078:
    raise RuntimeError("512x512 属性输出词表布局计算错误（bezier edge 扩展应为 8078）")
if TOKEN_LAYOUT.vocab_size != 8097:
    raise RuntimeError("512x512 输入词表布局计算错误（bezier edge 扩展应为 8097）")
