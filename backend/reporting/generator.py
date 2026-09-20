"""
Generates the deliverable explicitly required by the problem statement:
"a timestamped and structured lightweight text file of the conducted
steps with outcomes/status."

Produces two files per run:
  - reports/{run_id}.report.json  (machine-readable, matches Section 20's
    JSON/CSV logging requirement)
  - reports/{run_id}.report.txt   (human-readable summary — the
    "lightweight text file" called for explicitly)

Built from the DB (ProcedureState + Alert rows), not from memory, so a
report can be regenerated for any past run, not just one still in
process — matching Section 21's replay/explainability requirement.
"""
from __future__ import annotations
import os
import json
import datetime as dt
from sqlalchemy.orm import Session

from backend.db import models

REPORTS_DIR = os.environ.get("ASTRA_REPORTS_DIR", "reports")


def _fmt_ts(ts: float) -> str:
    return dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")


def generate_report(db: Session, run_id: str, config) -> dict:
    """
    Builds and writes the report for a completed (or in-progress) run.
    Returns the summary dict that also gets stored in the DB Report row.
    `config` is the ExperimentConfig the run was executed against, used to
    label steps that were never reached as SKIPPED/NOT_REACHED rather than
    silently omitting them.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    run = db.query(models.ExperimentRun).get(run_id)
    states = (
        db.query(models.ProcedureState)
        .filter_by(run_id=run_id)
        .order_by(models.ProcedureState.ts)
        .all()
    )
    alerts = (
        db.query(models.Alert)
        .filter_by(run_id=run_id)
        .order_by(models.Alert.ts)
        .all()
    )

    completed_step_ids = {s.step_id: s for s in states if s.status == "COMPLETED"}
    skipped_step_ids = {s.step_id: s for s in states if s.status == "SKIPPED"}

    step_rows = []
    for step in config.steps:
        if step.id in completed_step_ids:
            s = completed_step_ids[step.id]
            step_rows.append({
                "step_id": step.id, "name": step.name or step.id, "status": "COMPLETED",
                "completed_at": _fmt_ts(s.ts), "completed_at_unix": s.ts,
            })
        elif step.id in skipped_step_ids:
            s = skipped_step_ids[step.id]
            step_rows.append({
                "step_id": step.id, "name": step.name or step.id, "status": "SKIPPED",
                "completed_at": _fmt_ts(s.ts), "completed_at_unix": s.ts,
            })
        else:
            step_rows.append({
                "step_id": step.id, "name": step.name or step.id, "status": "NOT_REACHED",
                "completed_at": None, "completed_at_unix": None,
            })

    alert_rows = [{
        "ts": a.ts,
        "timestamp": _fmt_ts(a.ts),
        "type": a.type,
        "step_id": a.step_id,
        "severity": a.severity,
        "status": a.status,
    } for a in alerts]

    # Section 37: "most important — PROCEDURE COMPLIANCE ACCURACY".
    # Deliberately simple and auditable: fraction of required (non-optional)
    # steps completed without ever accumulating a CONFIRMED deviation of
    # any kind (SKIPPED/WRONG_ORDER/REPEATED/WRONG_OBJECT/FAILED/UNCERTAIN
    # against that step). TIMEOUT retries that self-resolved don't count
    # against compliance; unresolved ones do, since the step never
    # completed clean.
    flagged_step_ids = {a.step_id for a in alerts if a.step_id and a.status != "RECOVERED"}
    required_steps = [s for s in config.steps if not s.optional]
    clean_required = [
        s for s in required_steps
        if s.id in completed_step_ids and s.id not in flagged_step_ids
    ]
    compliance_accuracy = (len(clean_required) / len(required_steps)) if required_steps else None

    alert_type_counts: dict[str, int] = {}
    for a in alerts:
        alert_type_counts[a.type] = alert_type_counts.get(a.type, 0) + 1

    overall_status = run.status if run else "UNKNOWN"

    summary = {
        "run_id": run_id,
        "experiment": config.experiment,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "overall_status": overall_status,
        "started_at": run.started_at.isoformat() if run and run.started_at else None,
        "ended_at": run.ended_at.isoformat() if run and run.ended_at else None,
        "steps": step_rows,
        "steps_completed": len(completed_step_ids),
        "steps_skipped": len(skipped_step_ids),
        "steps_total": len(config.steps),
        "alerts": alert_rows,
        "alert_count": len(alert_rows),
        "alert_type_counts": alert_type_counts,
        "procedure_compliance_accuracy": compliance_accuracy,
    }

    json_path = os.path.join(REPORTS_DIR, f"{run_id}.report.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    txt_path = os.path.join(REPORTS_DIR, f"{run_id}.report.txt")
    with open(txt_path, "w") as f:
        f.write(f"ASTRA Experiment Report\n")
        f.write(f"{'=' * 40}\n")
        f.write(f"Experiment : {config.experiment}\n")
        f.write(f"Run ID     : {run_id}\n")
        f.write(f"Status     : {overall_status}\n")
        f.write(f"Started    : {run.started_at if run else '—'}\n")
        f.write(f"Ended      : {run.ended_at if run else '—'}\n")
        f.write(f"Steps      : {len(completed_step_ids)}/{len(config.steps)} completed"
                f" ({len(skipped_step_ids)} skipped)\n")
        if compliance_accuracy is not None:
            f.write(f"Procedure compliance accuracy: {compliance_accuracy * 100:.1f}%"
                    f" ({len(clean_required)}/{len(required_steps)} required steps clean)\n")
        f.write(f"\nStep-by-step:\n{'-' * 40}\n")
        for row in step_rows:
            mark = {"COMPLETED": "[x]", "SKIPPED": "[s]"}.get(row["status"], "[ ]")
            when = f" — {row['completed_at']}" if row["completed_at"] else ""
            f.write(f"{mark} {row['step_id']}: {row['name']} [{row['status']}]{when}\n")
        f.write(f"\nAlerts ({len(alert_rows)}):\n{'-' * 40}\n")
        if not alert_rows:
            f.write("None.\n")
        for a in alert_rows:
            f.write(f"[{a['timestamp']}] {a['type']} ({a['severity']}) near step {a['step_id']} (status: {a['status']})\n")
        if alert_type_counts:
            f.write(f"\nAlert breakdown: " + ", ".join(f"{k}={v}" for k, v in sorted(alert_type_counts.items())) + "\n")

    report_row = models.Report(
        run_id=run_id,
        summary_json=summary,
        file_ref=json_path,
    )
    db.add(report_row)
    db.commit()

    return summary
