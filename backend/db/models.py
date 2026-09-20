import uuid
import datetime as dt
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, ForeignKey, JSON, Text
)
from sqlalchemy.orm import relationship
from .database import Base


def _uuid():
    return str(uuid.uuid4())


def _now():
    return dt.datetime.utcnow()


class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    config_path = Column(String, nullable=False)   # path to YAML definition
    created_at = Column(DateTime, default=_now)

    runs = relationship("ExperimentRun", back_populates="experiment")


class Participant(Base):
    __tablename__ = "participants"
    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)
    meta = Column(JSON, default=dict)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    id = Column(String, primary_key=True, default=_uuid)
    experiment_id = Column(String, ForeignKey("experiments.id"))
    participant_id = Column(String, ForeignKey("participants.id"), nullable=True)
    started_at = Column(DateTime, default=_now)
    ended_at = Column(DateTime, nullable=True)
    status = Column(String, default="RUNNING")   # RUNNING | COMPLETED | ABORTED | FAILED
    video_file_ref = Column(String, nullable=True)
    source_type = Column(String, default="webcam")  # webcam | upload

    experiment = relationship("Experiment", back_populates="runs")


class ObjectInstance(Base):
    __tablename__ = "objects"
    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("experiment_runs.id"))
    class_name = Column(String)
    object_key = Column(String)   # matches ObjectDef.id in the experiment config
    first_seen_ts = Column(Float)
    last_seen_ts = Column(Float)


class ObjectState(Base):
    __tablename__ = "object_states"
    id = Column(String, primary_key=True, default=_uuid)
    object_id = Column(String, ForeignKey("objects.id"))
    state = Column(String)
    ts = Column(Float)
    confidence = Column(Float, default=1.0)


class Detection(Base):
    __tablename__ = "detections"
    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("experiment_runs.id"))
    frame_ts = Column(Float)
    entity_type = Column(String)   # person | object
    entity_id = Column(String)
    bbox_json = Column(JSON)
    confidence = Column(Float)


class Pose(Base):
    __tablename__ = "poses"
    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("experiment_runs.id"))
    frame_ts = Column(Float)
    person_track_id = Column(String)
    joints_json = Column(JSON)
    frame_type = Column(String, default="2d")   # 2d | 3d


class Activity(Base):
    __tablename__ = "activities"
    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("experiment_runs.id"))
    start_ts = Column(Float)
    end_ts = Column(Float)
    label = Column(String)
    confidence = Column(Float)
    source_model = Column(String, default="stub")


class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("experiment_runs.id"))
    ts = Column(Float)
    hand_id = Column(String)
    object_id = Column(String, nullable=True)
    interaction_type = Column(String)
    confidence = Column(Float)


class ProcedureState(Base):
    __tablename__ = "procedure_states"
    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("experiment_runs.id"))
    ts = Column(Float)
    step_id = Column(String)
    status = Column(String)   # PENDING|TENTATIVE|CONFIRMED|COMPLETED|SKIPPED|WRONG_ORDER|...
    snapshot_json = Column(JSON)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("experiment_runs.id"))
    ts = Column(Float)
    type = Column(String)          # SKIPPED|WRONG_ORDER|REPEATED|WRONG_OBJECT|TIMEOUT|UNCERTAIN|...
    step_id = Column(String, nullable=True)
    severity = Column(String, default="WARNING")   # INFO|WARNING|ERROR
    evidence_json = Column(JSON, default=dict)
    status = Column(String, default="TENTATIVE")   # TENTATIVE|CONFIRMED|RECOVERED


class LogEntry(Base):
    __tablename__ = "logs"
    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("experiment_runs.id"))
    ts = Column(Float)
    level = Column(String, default="INFO")
    message = Column(Text)
    context_json = Column(JSON, default=dict)


class Report(Base):
    __tablename__ = "reports"
    id = Column(String, primary_key=True, default=_uuid)
    run_id = Column(String, ForeignKey("experiment_runs.id"))
    generated_at = Column(DateTime, default=_now)
    summary_json = Column(JSON, default=dict)
    file_ref = Column(String, nullable=True)
