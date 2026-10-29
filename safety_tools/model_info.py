#!/usr/bin/env python3
"""读取模型文件的 task 和类别名称，供 Web 后端在独立子进程里调用。

物资计数需要按模型自身的类别校验和渲染勾选项。Web 后端是常驻进程，
如果在服务进程里加载 PyTorch 会常驻数百 MB 内存，因此这里由后端以子进程
方式调用，只输出一行 JSON（ASCII 转义，避免 Windows 控制台编码问题）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


def read_model_info(path: Path) -> dict:
    info: dict = {"ok": False, "path": str(path), "task": "", "names": [], "error": ""}
    if not path.is_file():
        info["error"] = f"模型文件不存在: {path}"
        return info
    try:
        from ultralytics import YOLO

        model = YOLO(str(path), task=None, verbose=False)
        info["task"] = str(getattr(model, "task", "") or "").lower()
        names = getattr(model, "names", None) or {}
        if isinstance(names, dict):
            info["names"] = [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
        else:
            info["names"] = [str(name) for name in names]
        info["ok"] = True
    except Exception as exc:
        info["error"] = str(exc)
    return info


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(json.dumps(
            {"ok": False, "path": "", "task": "", "names": [], "error": "用法: model_info.py <模型路径>"},
            ensure_ascii=True,
        ))
        return 2
    info = read_model_info(Path(args[0]))
    print(json.dumps(info, ensure_ascii=True))
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
