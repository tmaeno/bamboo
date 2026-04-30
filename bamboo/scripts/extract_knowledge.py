"""Deprecated shim — use ``bamboo populate --dry-run`` instead.

This module is kept for backwards compatibility.  All functionality has been
merged into :mod:`bamboo.scripts.populate_knowledge`.
"""

import click


@click.command()
@click.option("--email-thread", type=click.Path(exists=True))
@click.option("--task-data", type=click.Path(exists=True), default=None)
@click.option("--task-id", type=int, default=None)
@click.option("--external-data", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option("--max-retries", type=click.IntRange(min=0), default=None)
@click.option("--require-procedures", is_flag=True, default=False)
def main(email_thread, task_data, task_id, external_data, output, verbose, max_retries, require_procedures):
    """Extract knowledge graph preview without writing to any database.

    Deprecated: use ``bamboo populate --dry-run`` instead.
    """
    from bamboo.scripts.populate_knowledge import main as _populate_main  # noqa: PLC0415

    ctx = click.get_current_context()
    ctx.invoke(
        _populate_main,
        email_thread=email_thread,
        task_data=task_data,
        task_id=task_id,
        external_data=external_data,
        require_procedures=require_procedures,
        dry_run=True,
        output=output,
        max_retries=max_retries,
        debug_report=None,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
