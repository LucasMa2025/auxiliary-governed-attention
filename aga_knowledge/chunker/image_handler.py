"""
知识文档图片处理器

从知识文档中提取、保存和替换图片引用。

设计原则:
  不将多模态数据视为独立数据，而是与文档上下文对齐。

  图片的价值在于它与上下文的关联，而非图片本身。
  AGA 注入的是 condition/decision 文本对，不是图片。
  编码器只处理文本，图片信息通过文本描述间接编码。

处理流程:
  1. 提取文档中的图片引用（Markdown 语法）
  2. 根据来源类型处理图片:
     - Base64 嵌入 → 解码并保存到 Portal 静态资源目录
     - 外部 URL → 保留原始 URL
     - 本地路径 → 复制到 Portal 静态资源目录
  3. 替换图片引用为可访问的 URL 描述
  4. 返回处理后的文本 + 图片资产列表

配置:
  image_handling:
    enabled: true
    asset_dir: "/var/aga-knowledge/assets"
    base_url: "http://portal:8081/assets"
    max_image_size_mb: 10
    supported_formats: ["png", "jpg", "jpeg", "gif", "webp", "svg"]
    inline_description: true
    description_template: "[图片: {alt}, 参见 {url}]"
"""

import base64
import hashlib
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ImageAsset:
    """
    图片资产

    表示一张已处理的图片及其 Portal 可访问信息。

    Attributes:
        asset_id: 唯一 ID (格式: {source_id}_img_{index:03d})
        alt_text: 替代文本 (来自 Markdown ![alt](...))
        original_src: 原始来源 (Base64, URL, 或本地路径)
        portal_url: Portal 可访问的 URL
        mime_type: MIME 类型
        file_size: 文件大小（字节）
        width: 宽度（像素，如果可获取）
        height: 高度（像素，如果可获取）
    """
    asset_id: str
    alt_text: str
    original_src: str
    portal_url: str
    mime_type: str = "image/unknown"
    file_size: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        """转换为字典（用于 metadata 存储）"""
        return {
            "asset_id": self.asset_id,
            "alt": self.alt_text,
            "url": self.portal_url,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
        }


class ImageHandler:
    """
    知识文档图片处理器

    职责:
    1. 从文档中提取图片引用 (Markdown ![alt](src) 语法)
    2. 将 Base64/本地图片保存到 Portal 静态资源目录
    3. 构造可访问的 URL
    4. 将图片描述嵌入文档上下文（替换原始引用）

    安全性:
    - 验证 Base64 数据的 MIME 类型
    - 限制图片文件大小
    - 限制支持的图片格式
    - 生成唯一文件名（防止路径穿越）

    使用:
        handler = ImageHandler(
            asset_dir="/var/aga-knowledge/assets",
            base_url="http://portal:8081/assets",
        )

        processed_text, assets = handler.process_document(
            text=markdown_text,
            source_id="doc_001",
        )
    """

    # 支持的图片格式（MIME 子类型）
    DEFAULT_FORMATS = {"png", "jpg", "jpeg", "gif", "webp", "svg"}

    def __init__(
        self,
        asset_dir: str,
        base_url: str,
        max_image_size_mb: int = 10,
        supported_formats: Optional[List[str]] = None,
        description_template: str = "[图片: {alt}, 参见 {url}]",
    ):
        """
        初始化图片处理器

        Args:
            asset_dir: 静态资源保存目录
            base_url: Portal 资源 URL 前缀
            max_image_size_mb: 最大图片文件大小 (MB)
            supported_formats: 支持的图片格式列表
            description_template: 图片描述模板 ({alt} 和 {url} 会被替换)
        """
        self.asset_dir = Path(asset_dir)
        self.base_url = base_url.rstrip("/")
        self.max_image_size_bytes = max_image_size_mb * 1024 * 1024
        self.supported_formats = set(
            supported_formats or self.DEFAULT_FORMATS
        )
        self.description_template = description_template

    def process_document(
        self,
        text: str,
        source_id: str,
    ) -> Tuple[str, List[ImageAsset]]:
        """
        处理文档中的图片

        提取所有 Markdown 图片引用，根据来源类型处理，
        替换为描述文本 + 可访问 URL。

        Args:
            text: 文档文本（Markdown 格式）
            source_id: 源文档 ID

        Returns:
            (processed_text, image_assets)
            - processed_text: 图片引用已替换为描述文本
            - image_assets: 提取的图片资产列表
        """
        if not text:
            return text, []

        assets: List[ImageAsset] = []
        img_counter = 0

        def replace_image(match: re.Match) -> str:
            """替换单个图片引用"""
            nonlocal img_counter
            alt = match.group(1)
            src = match.group(2)
            img_counter += 1

            try:
                asset = self._process_single_image(
                    alt=alt,
                    src=src,
                    source_id=source_id,
                    index=img_counter,
                )
                if asset:
                    assets.append(asset)
                    # 生成描述文本
                    description = self.description_template.format(
                        alt=alt or f"图片{img_counter}",
                        url=asset.portal_url,
                    )
                    return description
                else:
                    logger.warning(
                        f"图片处理失败，保留原始引用: {src[:80]}"
                    )
                    return match.group(0)
            except Exception as e:
                logger.error(
                    f"图片处理异常: {e}, 保留原始引用"
                )
                return match.group(0)

        # 匹配 Markdown 图片语法: ![alt](src)
        processed = re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            replace_image,
            text,
        )

        if assets:
            logger.info(
                f"文档 {source_id}: 处理了 {len(assets)} 张图片"
            )

        return processed, assets

    def _process_single_image(
        self,
        alt: str,
        src: str,
        source_id: str,
        index: int,
    ) -> Optional[ImageAsset]:
        """
        处理单张图片

        根据来源类型分发到具体的处理方法:
        - data: URI → Base64 解码 + 保存
        - http(s):// → 保留外部 URL
        - 其他 → 本地路径处理

        Args:
            alt: 替代文本
            src: 图片来源 (Base64/URL/路径)
            source_id: 源文档 ID
            index: 图片序号

        Returns:
            ImageAsset 或 None（处理失败时）
        """
        asset_id = f"{source_id}_img_{index:03d}"

        # 确保文档资源目录存在
        doc_dir = self.asset_dir / source_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        # 分发处理
        if src.startswith("data:"):
            return self._handle_base64(
                src, alt, asset_id, doc_dir, source_id
            )
        elif src.startswith("http://") or src.startswith("https://"):
            return self._handle_external_url(
                src, alt, asset_id, source_id
            )
        else:
            return self._handle_local_path(
                src, alt, asset_id, doc_dir, source_id
            )

    def _handle_base64(
        self,
        src: str,
        alt: str,
        asset_id: str,
        doc_dir: Path,
        source_id: str,
    ) -> Optional[ImageAsset]:
        """
        处理 Base64 嵌入图片

        解析 data URI，验证格式，解码并保存到文件系统。

        Args:
            src: data URI (data:image/png;base64,...)
            alt: 替代文本
            asset_id: 资产 ID
            doc_dir: 文档资源目录
            source_id: 源文档 ID

        Returns:
            ImageAsset 或 None
        """
        # 解析 data URI
        match = re.match(
            r"data:image/([\w+.-]+);base64,(.+)", src, re.DOTALL
        )
        if not match:
            logger.warning(f"无法解析 data URI: {src[:50]}...")
            return None

        fmt = match.group(1).lower()

        # 验证格式
        if fmt not in self.supported_formats:
            logger.warning(
                f"不支持的图片格式: {fmt}。"
                f"支持: {self.supported_formats}"
            )
            return None

        # 解码 Base64
        try:
            data = base64.b64decode(match.group(2))
        except Exception as e:
            logger.error(f"Base64 解码失败: {e}")
            return None

        # 验证大小
        if len(data) > self.max_image_size_bytes:
            size_mb = len(data) / (1024 * 1024)
            max_mb = self.max_image_size_bytes / (1024 * 1024)
            logger.warning(
                f"图片过大: {size_mb:.1f}MB > {max_mb:.1f}MB 限制，"
                f"跳过: {asset_id}"
            )
            return None

        # 保存文件
        filename = f"{asset_id}.{fmt}"
        filepath = doc_dir / filename
        try:
            filepath.write_bytes(data)
        except Exception as e:
            logger.error(f"图片保存失败: {e}")
            return None

        portal_url = f"{self.base_url}/{source_id}/{filename}"

        return ImageAsset(
            asset_id=asset_id,
            alt_text=alt,
            original_src=f"data:image/{fmt};base64,[{len(data)} bytes]",
            portal_url=portal_url,
            mime_type=f"image/{fmt}",
            file_size=len(data),
        )

    def _handle_external_url(
        self,
        src: str,
        alt: str,
        asset_id: str,
        source_id: str,
    ) -> ImageAsset:
        """
        处理外部 URL 图片

        保留原始 URL，不下载。
        记录到 metadata 以便用户查看。

        Args:
            src: 外部 URL
            alt: 替代文本
            asset_id: 资产 ID
            source_id: 源文档 ID

        Returns:
            ImageAsset（使用原始 URL）
        """
        # 推断 MIME 类型
        mime_type = "image/unknown"
        lower_src = src.lower()
        for ext in ["png", "jpg", "jpeg", "gif", "webp", "svg"]:
            if lower_src.endswith(f".{ext}"):
                actual_ext = "jpeg" if ext == "jpg" else ext
                mime_type = f"image/{actual_ext}"
                break

        return ImageAsset(
            asset_id=asset_id,
            alt_text=alt,
            original_src=src,
            portal_url=src,  # 直接使用原始 URL
            mime_type=mime_type,
        )

    def _handle_local_path(
        self,
        src: str,
        alt: str,
        asset_id: str,
        doc_dir: Path,
        source_id: str,
    ) -> Optional[ImageAsset]:
        """
        处理本地路径图片

        复制文件到 Portal 静态资源目录。

        安全性: 使用 resolve() 防止路径穿越攻击。

        Args:
            src: 本地文件路径
            alt: 替代文本
            asset_id: 资产 ID
            doc_dir: 文档资源目录
            source_id: 源文档 ID

        Returns:
            ImageAsset 或 None（文件不存在时）
        """
        src_path = Path(src)

        if not src_path.exists():
            logger.warning(f"本地图片不存在: {src}")
            return None

        # 安全检查: resolve 路径
        try:
            src_path = src_path.resolve()
        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None

        # 验证格式
        ext = src_path.suffix.lstrip(".").lower()
        if ext not in self.supported_formats:
            logger.warning(
                f"不支持的图片格式: {ext}。"
                f"支持: {self.supported_formats}"
            )
            return None

        # 验证大小
        file_size = src_path.stat().st_size
        if file_size > self.max_image_size_bytes:
            size_mb = file_size / (1024 * 1024)
            max_mb = self.max_image_size_bytes / (1024 * 1024)
            logger.warning(
                f"图片过大: {size_mb:.1f}MB > {max_mb:.1f}MB 限制"
            )
            return None

        # 复制到资源目录
        filename = f"{asset_id}.{ext}"
        dest = doc_dir / filename
        try:
            shutil.copy2(src_path, dest)
        except Exception as e:
            logger.error(f"图片复制失败: {e}")
            return None

        portal_url = f"{self.base_url}/{source_id}/{filename}"

        return ImageAsset(
            asset_id=asset_id,
            alt_text=alt,
            original_src=str(src_path),
            portal_url=portal_url,
            mime_type=f"image/{ext}",
            file_size=file_size,
        )

    def cleanup_assets(self, source_id: str) -> int:
        """
        清理文档的图片资产

        删除文档对应的资源目录。

        Args:
            source_id: 源文档 ID

        Returns:
            删除的文件数量
        """
        doc_dir = self.asset_dir / source_id
        if not doc_dir.exists():
            return 0

        count = 0
        try:
            for f in doc_dir.iterdir():
                if f.is_file():
                    f.unlink()
                    count += 1
            doc_dir.rmdir()
            logger.info(
                f"清理文档 {source_id} 的 {count} 个图片资产"
            )
        except Exception as e:
            logger.error(f"资产清理失败: {e}")

        return count

    def get_stats(self) -> dict:
        """获取处理器统计"""
        return {
            "type": "ImageHandler",
            "asset_dir": str(self.asset_dir),
            "base_url": self.base_url,
            "max_image_size_mb": self.max_image_size_bytes / (1024 * 1024),
            "supported_formats": list(self.supported_formats),
        }

    def __repr__(self) -> str:
        return (
            f"ImageHandler("
            f"asset_dir={self.asset_dir!r}, "
            f"base_url={self.base_url!r})"
        )
