"""Pydantic request schemas (validation happens before any logic runs)."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

Stage = Literal["sediment", "carbon", "ro"]
Sensor = Literal["dp_sediment_bar", "flow_Lpm", "out_tds_ppm", "feed_tds_ppm", "temp_C", "inlet_bar"]
Kind = Literal["stuck", "drift", "spike", "dropout", "tds_inversion"]


class FaultIn(BaseModel):
    kind: Kind
    sensor: Sensor = "dp_sediment_bar"
    start: int = Field(ge=0)
    duration: int = Field(default=20, ge=1, le=400)
    magnitude: float = Field(default=0.0, ge=-5, le=50, description="drift: signed fraction (0.3 = +30%); spike/inversion: multiplier")


class HouseholdIn(BaseModel):
    name: str = Field(default="", max_length=80)
    scenario: str = "normal"
    law: Literal["A", "B"] = "A"
    seed: int = Field(default=0, ge=0, le=10**9)
    sim_index: Optional[int] = Field(default=None, ge=0)
    overrides: dict[str, float | int] = Field(default_factory=dict)
    calibrated: bool = True
    now_day: int = Field(default=60, ge=0)
    user_faults: list[FaultIn] = Field(default_factory=list)


class AdvanceIn(BaseModel):
    days: int = Field(ge=1, le=365)
    step: int = Field(default=7, ge=1, le=30)
    model: Optional[str] = None


class ReplaceIn(BaseModel):
    stage: Stage
    note: Optional[str] = Field(default=None, max_length=200)


class WhatIfIn(BaseModel):
    stage: Stage = "sediment"
    overrides: dict[str, float | int] = Field(default_factory=dict)
    user_faults: list[FaultIn] = Field(default_factory=list)
    model: Optional[str] = None


class ConfirmIn(BaseModel):
    planned_day: Optional[int] = Field(default=None, ge=0)


class NoteIn(BaseModel):
    note: Optional[str] = Field(default=None, max_length=200)


class TrainIn(BaseModel):
    which: list[Literal["reference_v3", "person2"]] = ["reference_v3", "person2"]
    n: int = Field(default=120, ge=24, le=400)
    seed: int = Field(default=1000, ge=0)
    max_iter: int = Field(default=200, ge=20, le=600)


class IngestIn(BaseModel):
    name: str = Field(default="", max_length=80)
    rows: list[dict] = Field(min_length=15, description="daily rows: day, usage_L, and sensor readings (null allowed)")
    replacements: dict[Stage, list[int]] = Field(default_factory=dict, description="days each stage's filter was replaced (>0)")


class ValidationIn(BaseModel):
    n_households: int = Field(default=10, ge=4, le=60)
    seed: int = Field(default=2000, ge=0)
    methods: list[str] = Field(default_factory=lambda: ["trend", "volume_rule", "health_20pct"])
    calibrated: bool = True
    cost: Optional[dict[str, float | dict]] = None
