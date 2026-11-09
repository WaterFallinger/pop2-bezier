# Windows installer

Download **[PrimitivePainter-0.9.29-Setup.exe](https://github.com/wonderfulearth/primitive-operation-painter/releases/download/pop2-1b/PrimitivePainter-0.9.29-Setup.exe)** from the [POP 2 release](https://github.com/wonderfulearth/primitive-operation-painter/releases/tag/pop2-1b). Installer binaries are release attachments and are not tracked in Git.

The installer does not contain model weights. In Settings, download the public POP 2 model or select local `model.pt` and `caption_spm_1500.model` files. Public model downloads do not require a Hugging Face account; online setup needs access to the download hosts.

This is the existing tested 0.9.29 installer. The cleaned source candidate has not been used to reproduce that exact binary. See [binary provenance](../docs/binary_provenance.md).

Application source and build instructions are under [app/](../app/README.md). Training and Python inference are under [model/](../model/README.md). The image-to-sequence generator is under [data_tools/](../data_tools/image_to_sequence/README.md).
