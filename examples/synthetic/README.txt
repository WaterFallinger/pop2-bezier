纯合成最小数据示例 / Synthetic data example

images/smoke_000001.png 由程序绘制的渐变、矩形与椭圆构成，不来自外部训练集。
sequences/ 包含正式图像序列化器的单图真实输出：默认896总记录预算，本例在394条记录时三阶段全部完成。
preview/ 是从CSV独立回放的PNG，不读取源图绘制。

图像主干（image_name）为 smoke_000001；训练数据准备时用同名提示词记录关联。
本例用于验证格式、caption cache与训练加载链路，不用于评估真实画质或模型泛化。

可从仓库根目录回放：
python data_tools/image_to_sequence/tools/render_csv.py --csv-root examples/synthetic/sequences --output ./synthetic_preview --limit 1

重新生成请使用新输出目录，避免对随仓库提供的样例执行续跑写入：
cargo run --locked --release --manifest-path data_tools/image_to_sequence/Cargo.toml -- --input examples/synthetic/images --output ./synthetic_regenerated --images 1 --versions 1 --gpus 1 --batch-size 1 --dispatch-size 1 --seed 20261105 --mode mixed

重新生成的几何细节可能随GPU/驱动浮点行为有所不同；训练协议与区域配额保持一致。
