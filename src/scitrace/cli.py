"""命令行入口（SPEC §7 的命令表）。

## 两条纪律

1. **``--json`` 输出的必须是纯 JSON**。所有人类可读的排版（rich 的表格、面板、
   颜色）都走另一条分支。混入一个进度条就会让 ``stc ask --json | jq`` 直接失败，
   而脚本化使用正是 ``--json`` 存在的全部理由。
2. **退出码是接口的一部分**。0 表示"命令成功执行"——**包括拒答**：
   拒答是一个正常结果而非错误，把它变成非零退出码会让调用方无法区分
   "系统说不知道"与"系统坏了"。1 是运行错误，2 是用法错误。
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys

import anyio
from pathlib import Path

from scitrace import __version__
from scitrace.api import ask, build_index, index_status, load_services, search
from scitrace.config import Settings, SettingsError, load_settings, save_named, settings_path
from scitrace.domain.session import SessionStatus
from scitrace.service import SessionStore

logger = logging.getLogger(__name__)

__all__ = ["build_parser", "main"]

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="stc",
        description="scitrace —— 证据可溯源的科研文献问答系统",
    )
    parser.add_argument("--version", action="version", version=f"scitrace {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="提高日志详细程度")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="摄入语料并建立索引")
    index.add_argument("paths", nargs="+", type=Path, help="语料文件或目录")
    index.add_argument("--settings", default=None, help="命名配置（默认 default）")
    index.add_argument("--rebuild", action="store_true", help="忽略既有清单，全量重建")

    ask_parser = subparsers.add_parser("ask", help="回答一个问题")
    ask_parser.add_argument("question", help="问题正文")
    ask_parser.add_argument("--settings", default=None)
    ask_parser.add_argument(
        "--mode", choices=["agentic", "deterministic"], default="deterministic"
    )
    ask_parser.add_argument("--language", choices=["zh", "en"], default="zh")
    ask_parser.add_argument("--json", action="store_true", dest="as_json")

    search_parser = subparsers.add_parser("search", help="只做检索，查看证据")
    search_parser.add_argument("query", help="查询串")
    search_parser.add_argument("--settings", default=None)
    search_parser.add_argument("--k", type=int, default=None)
    search_parser.add_argument("--json", action="store_true", dest="as_json")

    status = subparsers.add_parser("status", help="显示索引状态")
    status.add_argument("--settings", default=None)
    status.add_argument("--json", action="store_true", dest="as_json")

    sessions = subparsers.add_parser("sessions", help="历史问答")
    sessions_sub = sessions.add_subparsers(dest="sessions_command", required=True)
    sessions_search = sessions_sub.add_parser("search", help="在历史答案中检索")
    sessions_search.add_argument("keyword")
    sessions_search.add_argument("--settings", default=None)
    sessions_search.add_argument("--json", action="store_true", dest="as_json")

    config = subparsers.add_parser("config", help="配置管理")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    for name, help_text in (
        ("show", "显示当前生效的配置"),
        ("save", "把当前配置写入命名 profile"),
        ("path", "显示 profile 文件路径"),
        ("init", "生成一份默认 profile"),
    ):
        sub = config_sub.add_parser(name, help=help_text)
        sub.add_argument("--settings", default=None)

    return parser


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING - min(verbosity, 2) * 10
    logging.basicConfig(level=max(level, logging.DEBUG), format="%(levelname)s %(name)s: %(message)s")


def _emit(payload: dict, *, as_json: bool, human: str) -> None:
    """按 ``--json`` 选择输出形式。

    刻意集中在一处：散落的 ``if as_json`` 迟早会有一条分支忘了走 JSON 路径，
    而那种缺陷只在脚本化使用时才暴露。
    """
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    elif human:
        print(human)


def _answer_payload(result, session_id: str) -> dict:  # noqa: ANN001
    """按 SPEC §7 的 schema 构造 ``ask`` 的 JSON 输出。"""
    answer = result.answer
    return {
        "session_id": session_id,
        "question": result.question,
        "answer": answer.text,
        "status": str(result.status),
        "citations": [
            {
                "evidence_key": item.evidence_key,
                "citation": item.inline,
                "source_key": item.source_key,
                "reference_key": item.reference_key,
                "page_label": item.page_label,
            }
            for item in answer.citations
        ],
        "references": answer.references,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "estimated_cost_usd": result.usage.estimated_cost_usd,
            "llm_calls": result.usage.llm_calls,
        },
        "timing": {
            "retrieve_s": result.timing.retrieve_s,
            "screen_s": result.timing.screen_s,
            "synthesize_s": result.timing.synthesize_s,
            "total_s": result.timing.total_s,
        },
        "notes": result.notes,
    }


def _render_answer(payload: dict) -> str:
    """人类可读的答案渲染。"""
    answer = payload["answer"] or "（无答案正文）"
    lines = [answer, ""]
    if payload["citations"]:
        lines.append("参考文献：")
        for key, entry in payload["references"].items():
            first_line = entry.splitlines()[0] if entry else ""
            lines.append(f"  [{key}] {first_line}")
    else:
        lines.append("（没有可回溯的引用）")
    usage = payload["usage"]
    lines.append(
        f"\n状态：{payload['status']}　耗时 {payload['timing']['total_s']:.1f}s　"
        f"token {usage['prompt_tokens'] + usage['completion_tokens']}　"
        f"成本 ${usage['estimated_cost_usd']:.4f}"
    )
    if payload["notes"]:
        lines.append(f"备注：{', '.join(payload['notes'])}")
    return "\n".join(lines)


def _load(name: str | None) -> Settings:
    return load_settings(name=name)


async def _cmd_index(args: argparse.Namespace) -> int:
    settings = _load(args.settings)
    services = load_services(settings, load_index=not args.rebuild)
    try:
        report = await build_index(services, args.paths, rebuild=args.rebuild)
    finally:
        await services.aclose()

    print(report.summary())
    for identifier, error in report.failed:
        print(f"  失败：{identifier} —— {error}", file=sys.stderr)
    if report.duplicate_sources:
        for key, paths in report.duplicate_sources.items():
            print(f"  注意：{paths} 归并为同一篇文献（{key}）", file=sys.stderr)
    # 全部失败才算命令失败：部分失败已经在上面的报告里给出，退出码仍为 0，
    # 否则脚本化的批量摄入会因为一个坏文件而中断后续处理。
    return EXIT_ERROR if (report.failed and not report.added and not report.updated) else EXIT_OK


async def _cmd_ask(args: argparse.Namespace) -> int:
    settings = _load(args.settings)
    services = load_services(settings, language=args.language)
    try:
        result = await ask(args.question, services, mode=args.mode)
    finally:
        await services.aclose()

    payload = _answer_payload(result, getattr(result, "session_id", ""))
    _emit(payload, as_json=args.as_json, human=_render_answer(payload))
    # 拒答是正常终态，退出码仍为 0（SPEC §7）
    return EXIT_OK


async def _cmd_search(args: argparse.Namespace) -> int:
    settings = _load(args.settings)
    services = load_services(settings)
    try:
        results = await search(args.query, services, k=args.k)
    finally:
        await services.aclose()

    payload = {
        "query": args.query,
        "results": [
            {
                "fragment_id": item.fragment.fragment_id,
                "source_key": item.fragment.source_key,
                "score": item.score,
                "origin": item.origin,
                "rank": item.rank,
                "section_path": item.fragment.section_path,
                "page_label": item.fragment.page_label,
                "text": item.fragment.text,
            }
            for item in results
        ],
    }
    human = "\n\n".join(
        f"[{index}] {item.fragment.page_label or '页码未知'} "
        f"({item.origin} {item.score:.4f})\n{item.fragment.text[:300]}"
        for index, item in enumerate(results, start=1)
    ) or "没有检索到结果。"
    _emit(payload, as_json=args.as_json, human=human)
    return EXIT_OK


def _cmd_status(args: argparse.Namespace) -> int:
    settings = _load(args.settings)
    payload = index_status(settings)
    human = "\n".join(
        [
            f"索引指纹：{payload['fingerprint']}",
            f"索引目录：{payload['index_dir']}",
            f"文献：{payload['sources']}　片段：{payload['fragments']}　失败文件：{payload['failed']}",
            "语料："
            + (
                "，".join(f"{root}（{count}）" for root, count in payload["corpus"].items())
                or "（空）"
            ),
        ]
    )
    _emit(payload, as_json=args.as_json, human=human)
    return EXIT_OK


def _cmd_sessions(args: argparse.Namespace) -> int:
    settings = _load(args.settings)
    sessions = SessionStore(settings.sessions_dir).load_sessions()
    keyword = args.keyword.lower()
    hits = [
        {
            "session_id": session.session_id,
            "question": session.question,
            "status": str(session.status),
            "answer": session.answer.text if session.answer else "",
        }
        for session in sessions.values()
        if keyword in session.question.lower()
        or (session.answer is not None and keyword in session.answer.text.lower())
    ]
    payload = {"keyword": args.keyword, "count": len(hits), "sessions": hits}
    human = "\n\n".join(
        f"[{item['status']}] {item['question']}\n{item['answer'][:200]}" for item in hits
    ) or "没有匹配的历史问答。"
    _emit(payload, as_json=args.as_json, human=human)
    return EXIT_OK


def _cmd_config(args: argparse.Namespace) -> int:
    settings = _load(args.settings)
    name = args.settings or "default"
    if args.config_command == "show":
        payload = settings.to_payload()
        _emit(payload, as_json=True, human="")
        return EXIT_OK
    if args.config_command == "path":
        print(settings_path(name))
        return EXIT_OK
    if args.config_command in {"save", "init"}:
        written = save_named(settings, name)
        print(f"已写入 {written}")
        return EXIT_OK
    print(f"未知的 config 子命令：{args.config_command}", file=sys.stderr)
    return EXIT_USAGE


def main(argv: list[str] | None = None) -> int:
    """命令行主入口。

    Returns:
        进程退出码（SPEC §7）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    handlers = {
        "index": _cmd_index,
        "ask": _cmd_ask,
        "search": _cmd_search,
        "status": _cmd_status,
        "sessions": _cmd_sessions,
        "config": _cmd_config,
    }
    handler = handlers.get(args.command)
    if handler is None:  # pragma: no cover - argparse 已保证
        parser.print_usage(sys.stderr)
        return EXIT_USAGE

    try:
        # 命令分为同步（status/sessions/config）与异步（index/ask/search）两类。
        # 用 inspect 判断而不是把全部命令都写成 async：状态查询不碰网络，
        # 为它建一个事件循环是纯粹的浪费，也会让不依赖 anyio 的调用方被迫引入它。
        if inspect.iscoroutinefunction(handler):
            return anyio.run(handler, args)
        return handler(args)
    except SettingsError as error:
        print(f"配置错误：{error}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("已中断。", file=sys.stderr)
        return EXIT_ERROR
    except Exception as error:  # noqa: BLE001 - 不把 traceback 抛给用户
        logger.debug("命令执行失败", exc_info=True)
        print(f"执行失败：{error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
