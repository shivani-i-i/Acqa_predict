"""AquaPredict dashboard (Streamlit). Run from the repo root:   streamlit run aquapredict/app/ui/streamlit_app.py
Every control calls the real service layer (simulator, forecasters, rule checks, SQLite). Nothing is mocked."""
from __future__ import annotations
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:                      # lets `streamlit run` work without installing the package
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from aquapredict.app.service import AquaService, NotFound, BadRequest, Conflict, STAGE_LABELS
from aquapredict.app.registry import ModelUnavailable, DISCLAIMER
from aquapredict.planner.cost import CostConfig
from aquapredict.schema import STAGES

PAGES = ["Dashboard", "Filter detail", "Simulator & what-if", "Alerts", "Planner & costs", "Validation lab", "Models", "Data & export"]
SENSOR_OPTIONS = ["dp_sediment_bar", "flow_Lpm", "out_tds_ppm", "feed_tds_ppm", "temp_C", "inlet_bar"]
BADGE = {"ok": "🟢 OK", "warning": "🟠 warning", "fault_suspected": "🔴 fault suspected"}
st.set_page_config(page_title="AquaPredict", page_icon="💧", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --ink: #eef4ff;
  --muted: #9aa8bd;
  --aqua: #43d9d2;
  --aqua-dark: #8af1e9;
  --aqua-soft: #123737;
  --navy: #f4f7ff;
  --line: rgba(255,255,255,.10);
  --surface: rgba(20, 28, 42, .72);
  --canvas: #0b0f17;
  --indigo: #8b7dff;
  --amber: #f3bb63;
}

html, body {
  font-family: 'DM Sans', sans-serif;
}

section[data-testid="stMain"],
section[data-testid="stMain"] p,
section[data-testid="stMain"] label,
section[data-testid="stMain"] li,
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] {
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(800px 400px at 85% -5%, rgba(67, 217, 210, .08), transparent 70%),
    radial-gradient(700px 500px at 5% 90%, rgba(139, 125, 255, .07), transparent 70%),
    var(--canvas);
  color: var(--ink);
}

[data-testid="stHeader"] {
  background: rgba(11,15,23,.84);
  backdrop-filter: blur(14px);
}

[data-testid="stSidebar"] {
  display: none;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
  color: #e9f8fb;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
  background: rgba(255,255,255,.10);
  border-color: rgba(255,255,255,.20);
}

[data-testid="stSidebar"] [role="radiogroup"] label {
  border-radius: 10px;
  padding: .36rem .55rem;
  transition: background .2s ease, transform .2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(255,255,255,.10);
  transform: translateX(2px);
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: linear-gradient(90deg, rgba(33, 211, 199, .28), rgba(255,255,255,.08));
  box-shadow: inset 3px 0 0 #53e3d6;
}

[data-testid="stRadio"] > div[role="radiogroup"] {
  gap: .45rem;
}

section[data-testid="stMain"] [data-testid="stRadio"] label {
  min-height: 2rem;
  padding: .38rem .8rem;
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 999px;
  background: rgba(255,255,255,.04);
  color: #aebdd0;
  transition: all .2s ease;
}

section[data-testid="stMain"] [data-testid="stRadio"] label:hover {
  border-color: rgba(0,242,254,.35);
  background: rgba(0,242,254,.08);
  transform: translateY(-1px);
}

section[data-testid="stMain"] [data-testid="stRadio"] label:has(input:checked) {
  border-color: rgba(0,242,254,.55);
  background: linear-gradient(120deg, rgba(0,242,254,.16), rgba(0,201,167,.12));
  box-shadow: 0 0 18px rgba(0,242,254,.10);
  color: #e9feff;
}

section[data-testid="stMain"] [data-testid="stRadio"] label > div:first-child {
  display: none;
}

section[data-testid="stMain"] [data-testid="stRadioOption"] > div > div:first-child,
section[data-testid="stMain"] [data-testid="stRadioOption"] > div > div > div:first-child {
  display: none !important;
}

section[data-testid="stMain"] label[data-testid="stRadioOption"] > div > div:first-child {
  display: none !important;
}

section[data-testid="stMain"] label[data-testid="stRadioOption"] > div {
  padding-left: 0 !important;
}

section[data-testid="stMain"] [data-testid="stRadioOption"] > div {
  padding-left: 0 !important;
}

.ap-commandbar {
  position: sticky;
  top: .65rem;
  z-index: 20;
  margin: .15rem 0 .75rem;
  padding: .45rem .7rem;
  border: 1px solid rgba(56,189,248,.16);
  border-radius: 20px;
  background: rgba(14,22,38,.78);
  box-shadow: 0 18px 50px rgba(0,0,0,.26), inset 0 1px rgba(255,255,255,.05);
  backdrop-filter: blur(24px);
}

section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) {
  position: sticky;
  top: .65rem;
  z-index: 20;
  margin: .15rem 0 .75rem;
  padding: .45rem .7rem;
  border: 1px solid rgba(56,189,248,.16);
  border-radius: 20px;
  background: rgba(14,22,38,.78);
  box-shadow: 0 18px 50px rgba(0,0,0,.26), inset 0 1px rgba(255,255,255,.05);
  backdrop-filter: blur(24px);
}

section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) > [data-testid="column"] {
  align-items: center;
  min-width: 0;
}

section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) [data-testid="stSelectbox"] label {
  display: none;
}

section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) [data-testid="stRadio"] {
  min-width: 0;
}

section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) [data-testid="stRadio"] > div[role="radiogroup"] {
  justify-content: center;
  flex-wrap: nowrap;
  gap: .18rem;
}

section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) [data-testid="stRadio"] label {
  min-width: max-content;
  min-height: 2rem;
  padding: .3rem .62rem;
  border: 1px solid transparent;
  border-radius: 999px;
  color: #aebdd0;
  background: transparent;
}

section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) [data-testid="stRadio"] label > div:first-child,
section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) [data-testid="stRadio"] input {
  display: none !important;
}

section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) [data-testid="stRadioOption"] > div > div:first-child {
  display: none !important;
}

section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) [data-testid="stRadioOption"] > div {
  padding-left: 0 !important;
}

section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) [data-testid="stRadio"] label:has(input:checked) {
  border-color: rgba(0,242,254,.38);
  color: #effcff;
  background: rgba(0,242,254,.12);
  box-shadow: 0 0 16px rgba(0,242,254,.08);
}

@media (max-width: 1100px) {
  section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) {
    overflow-x: auto;
  }

  section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) > [data-testid="column"] {
    flex: 0 0 auto;
  }

  section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) > [data-testid="column"]:nth-child(1) {
    min-width: 17rem;
  }

  section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) > [data-testid="column"]:nth-child(2) {
    min-width: 31rem;
  }

  section[data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ap-commandbar__brand) > [data-testid="column"]:nth-child(3) {
    min-width: 22rem;
  }
}

.ap-commandbar__brand {
  display: flex;
  align-items: center;
  gap: .55rem;
  height: 2.2rem;
  color: #effcff;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  white-space: nowrap;
}

.ap-topnav [data-testid="stRadio"] > div[role="radiogroup"] {
  justify-content: center;
  flex-wrap: nowrap;
}

.ap-topnav [data-testid="stRadio"] label {
  min-height: 2rem;
  padding: .3rem .72rem;
  border: 1px solid transparent;
  border-radius: 999px;
  color: #aebdd0;
  background: transparent;
}

.ap-topnav [data-testid="stRadio"] label:hover {
  border-color: rgba(0,242,254,.25);
  background: rgba(0,242,254,.07);
}

.ap-topnav [data-testid="stRadio"] label:has(input:checked) {
  border-color: rgba(0,242,254,.38);
  color: #effcff;
  background: rgba(0,242,254,.12);
  box-shadow: 0 0 16px rgba(0,242,254,.08);
}

.ap-topnav [data-testid="stRadio"] label > div:first-child {
  display: none;
}

.ap-quiet-footer {
  margin-top: 2rem;
  padding: .75rem 0 1.5rem;
  border-top: 1px solid rgba(255,255,255,.07);
  color: #718198;
  font-size: .72rem;
  text-align: center;
}

.ap-commandbar__brand span {
  color: #6de8e0;
  font-size: .62rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
}

.ap-commandbar__brand i {
  display: grid;
  width: 1.9rem;
  height: 1.9rem;
  place-items: center;
  border: 1px solid rgba(0,242,254,.35);
  border-radius: .65rem;
  background: linear-gradient(135deg, rgba(0,242,254,.2), rgba(0,201,167,.12));
  box-shadow: 0 0 18px rgba(0,242,254,.12);
  font-style: normal;
}

.ap-commandbar [data-testid="stHorizontalBlock"] {
  align-items: center;
}

.ap-commandbar [data-testid="stSelectbox"] label,
.ap-commandbar [data-testid="stRadio"] > label {
  display: none;
}

.ap-commandbar [data-baseweb="select"] > div {
  min-height: 2.4rem;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 999px;
  background: rgba(255,255,255,.05);
}

.ap-commandbar [data-testid="stButton"] > button {
  min-height: 2.4rem;
  border-radius: 999px;
  font-size: .74rem;
}

h1, h2, h3, h4 {
  font-family: 'Space Grotesk', sans-serif;
  color: var(--navy) !important;
  letter-spacing: -.025em;
}

h1 { font-size: clamp(1.8rem, 3vw, 2.45rem); }
h2 { font-size: 1.55rem; }
h3 { font-size: 1.12rem; }

[data-testid="stMetric"] {
  background: rgba(20, 28, 42, .72);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  box-shadow: 0 18px 50px rgba(0,0,0,.18), inset 0 1px rgba(255,255,255,.05);
  backdrop-filter: blur(16px);
  transition: transform .28s cubic-bezier(.2,.8,.2,1), border-color .28s ease, box-shadow .28s ease;
}

[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  border-color: rgba(67,217,210,.35);
  box-shadow: 0 24px 60px rgba(0,0,0,.28), 0 0 30px rgba(67,217,210,.06);
}

[data-testid="stMetricLabel"] p {
  color: var(--muted);
  font-size: .78rem;
  font-weight: 600;
  letter-spacing: .02em;
}

[data-testid="stMetricValue"] {
  color: var(--navy) !important;
  font-family: 'Space Grotesk', sans-serif;
}

section[data-testid="stMain"] [data-testid="stInfo"] p,
section[data-testid="stMain"] [data-testid="stSuccess"] p,
section[data-testid="stMain"] [data-testid="stWarning"] p,
section[data-testid="stMain"] [data-testid="stError"] p {
  color: #172033 !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
  background: linear-gradient(145deg, rgba(28,38,55,.86), rgba(15,21,32,.78));
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: 0 18px 55px rgba(0,0,0,.22), inset 0 1px rgba(255,255,255,.04);
  backdrop-filter: blur(18px);
  transition: transform .28s cubic-bezier(.2,.8,.2,1), box-shadow .28s ease, border-color .28s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: rgba(67,217,210,.35);
  box-shadow: 0 28px 70px rgba(0,0,0,.30), 0 0 36px rgba(67,217,210,.07);
  transform: translateY(-2px);
}

.ap-hero {
  position: relative;
  overflow: hidden;
  margin: .2rem 0 1.35rem;
  padding: 1.65rem 1.8rem;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 26px;
  color: white;
  background:
    radial-gradient(circle at 88% 18%, rgba(67,217,210,.24), transparent 15rem),
    radial-gradient(circle at 60% 100%, rgba(139,125,255,.22), transparent 16rem),
    linear-gradient(120deg, #172236 0%, #142d3c 56%, #154d55 100%);
  box-shadow: 0 24px 70px rgba(0,0,0,.32), inset 0 1px rgba(255,255,255,.08);
}

.ap-hero:after {
  content: '';
  position: absolute;
  width: 15rem;
  height: 15rem;
  right: -5rem;
  bottom: -9rem;
  border: 1px solid rgba(255,255,255,.22);
  border-radius: 50%;
  box-shadow: 0 0 0 1.1rem rgba(255,255,255,.08), 0 0 0 2.2rem rgba(255,255,255,.05);
}

.ap-hero h1 {
  position: relative;
  z-index: 1;
  margin: 0;
  color: white !important;
  font-size: clamp(1.7rem, 3vw, 2.35rem);
}

.ap-hero p {
  position: relative;
  z-index: 1;
  max-width: 42rem;
  margin: .55rem 0 0;
  color: rgba(255,255,255,.78);
  font-size: .96rem;
}

.ap-kicker {
  color: var(--aqua);
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
}

.ap-note {
  margin: .9rem 0 1.2rem;
  padding: .75rem 1rem;
  border: 1px solid rgba(67,217,210,.24);
  border-left: 4px solid var(--aqua);
  border-radius: 0 12px 12px 0;
  background: rgba(67,217,210,.08);
  color: #c8d8e8;
}

.stButton > button {
  border: 1px solid rgba(67, 217, 210, .28);
  border-radius: 10px;
  color: var(--aqua-dark);
  background: rgba(255,255,255,.05);
  font-weight: 600;
  transition: all .2s ease;
}

.stButton > button:hover {
  border-color: var(--aqua);
  color: white;
  background: linear-gradient(135deg, #36c8c0, #6e63e9);
  box-shadow: 0 12px 26px rgba(67,217,210,.18);
  transform: translateY(-1px);
}

.stTabs [data-baseweb="tab-list"] {
  gap: .35rem;
  border-bottom: 1px solid var(--line);
}

.stTabs [data-baseweb="tab"] {
  padding: .65rem .9rem;
  border-radius: 9px 9px 0 0;
}

.stTabs [aria-selected="true"] {
  color: var(--aqua);
  background: rgba(67,217,210,.10);
}

.stAlert {
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,.10);
  background: rgba(35,42,55,.86);
}

section[data-testid="stMain"] [data-testid="stWarning"] {
  display: inline-flex;
  width: auto;
  margin: .25rem 0 1.1rem;
  padding: .55rem .85rem;
  border-color: rgba(243,187,99,.30);
  border-radius: 999px;
  background: rgba(243,187,99,.10);
}

section[data-testid="stMain"] [data-testid="stInfo"] {
  border-color: rgba(67,217,210,.25);
  border-radius: 999px;
  background: rgba(67,217,210,.09);
}

section[data-testid="stMain"] [data-testid="stSuccess"] {
  background: rgba(74,204,143,.10);
  border-color: rgba(74,204,143,.28);
}

section[data-testid="stMain"] [data-testid="stError"] {
  background: rgba(255,107,117,.10);
  border-color: rgba(255,107,117,.28);
}

section[data-testid="stMain"] [data-testid="stAlert"] * {
  color: #e9f1fb !important;
}

section[data-testid="stMain"] [data-testid="stWarning"] * {
  color: #ffe2aa !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
  color: #ffe2aa !important;
}

.ap-section-label {
  margin: 1.35rem 0 .65rem;
  color: #73849d;
  font-size: .7rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
}

.ap-ai-badge {
  display: inline-flex;
  align-items: center;
  gap: .45rem;
  padding: .35rem .65rem;
  border: 1px solid rgba(201,211,224,.22);
  border-radius: 999px;
  color: #c7d2df;
  background: rgba(201,211,224,.07);
  font-size: .72rem;
  font-weight: 700;
}

.ap-ai-badge:before {
  width: .42rem;
  height: .42rem;
  border-radius: 50%;
  background: #86c9bd;
  box-shadow: none;
  content: '';
}

.ap-bento {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: .75rem;
  margin: .25rem 0 1.1rem;
}

.ap-tile {
  min-height: 7rem;
  padding: 1rem 1.1rem;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 18px;
  background: linear-gradient(145deg, rgba(22,31,46,.86), rgba(13,19,30,.78));
  box-shadow: 0 16px 42px rgba(0,0,0,.18), inset 0 1px rgba(255,255,255,.04);
  backdrop-filter: blur(18px);
  transition: transform .28s cubic-bezier(.2,.8,.2,1), border-color .28s ease;
}

.ap-tile:hover {
  transform: translateY(-4px);
  border-color: rgba(67,217,210,.32);
}

.ap-tile--hero {
  grid-column: span 2;
  grid-row: span 2;
  min-height: 15rem;
  background:
    radial-gradient(circle at 85% 22%, rgba(67,217,210,.20), transparent 9rem),
    linear-gradient(145deg, rgba(24,56,66,.92), rgba(16,25,40,.90));
}

.ap-tile--wide { grid-column: span 2; }

.ap-tile__label {
  color: #8d9db3;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.ap-tile__value {
  margin-top: .55rem;
  color: #f2f6ff;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2rem;
  font-weight: 600;
  letter-spacing: -.05em;
}

.ap-tile__unit {
  color: #8d9db3;
  font-family: 'DM Sans', sans-serif;
  font-size: .78rem;
  letter-spacing: 0;
}

.ap-tile__meta {
  margin-top: .5rem;
  color: #a9b7c9;
  font-size: .78rem;
}

.ap-pulse {
  display: inline-flex;
  align-items: center;
  gap: .5rem;
  color: #86f2cd;
  font-size: .78rem;
  font-weight: 700;
}

.ap-pulse:before {
  width: .5rem;
  height: .5rem;
  border: 3px solid rgba(134,242,205,.22);
  border-radius: 50%;
  background: #86f2cd;
  box-shadow: 0 0 0 5px rgba(134,242,205,.08), 0 0 18px #86f2cd;
  content: '';
}

.ap-progress {
  height: .35rem;
  margin-top: 1rem;
  overflow: hidden;
  border-radius: 99px;
  background: rgba(255,255,255,.10);
}

.ap-progress span {
  display: block;
  width: 68%;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #00f2fe, #4facfe);
  box-shadow: 0 0 16px rgba(0,242,254,.35);
}

.ap-spatial {
  position: relative;
  min-height: 27rem;
  overflow: hidden;
  border: 1px solid rgba(56,189,248,.16);
  border-radius: 26px;
  background:
    radial-gradient(circle at 54% 48%, rgba(122,164,177,.10), transparent 12rem),
    linear-gradient(145deg, rgba(20,28,38,.94), rgba(9,14,21,.94));
  box-shadow: 0 24px 70px rgba(0,0,0,.32), inset 0 1px rgba(255,255,255,.06);
  backdrop-filter: blur(20px);
  perspective: 1000px;
}

.ap-spatial__grid {
  position: absolute;
  inset: auto -10% -38% -10%;
  height: 65%;
  transform: rotateX(68deg);
  background-image: linear-gradient(rgba(146,169,181,.10) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(146,169,181,.10) 1px, transparent 1px);
  background-size: 2rem 2rem;
  mask-image: linear-gradient(to top, black, transparent);
}

.ap-spatial__title {
  position: absolute;
  top: 1.25rem;
  left: 1.35rem;
  z-index: 4;
  color: #b7c8dc;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
}

.ap-twin {
  position: absolute;
  top: 4.1rem;
  left: 50%;
  width: 12rem;
  height: 19rem;
  transform: translateX(-50%) rotateY(-12deg) rotateX(4deg);
  transform-style: preserve-3d;
  filter: drop-shadow(0 1.8rem 1.2rem rgba(0,0,0,.32));
}

.ap-twin:before {
  position: absolute;
  top: 1.6rem;
  right: -.65rem;
  width: 1rem;
  height: 15.6rem;
  border: 1px solid rgba(200,214,218,.22);
  border-left: 0;
  border-radius: 0 .7rem .7rem 0;
  background: linear-gradient(90deg, #1d2b32, #0d171d);
  box-shadow: inset -.3rem 0 .6rem rgba(0,0,0,.28);
  content: '';
  transform: translateZ(-.35rem) skewY(-5deg);
}

.ap-twin:after {
  position: absolute;
  right: 1.6rem;
  bottom: -.9rem;
  left: 1.6rem;
  height: 1.1rem;
  border-radius: 50%;
  background: radial-gradient(ellipse, rgba(0,0,0,.45), transparent 70%);
  content: '';
  transform: rotateX(70deg);
}

.ap-twin__body {
  position: absolute;
  inset: 0;
  border: 1px solid rgba(191,205,214,.34);
  border-radius: 1.8rem 1.8rem .9rem .9rem;
  background: linear-gradient(110deg, #aeb9bd 0%, #65747c 18%, #26333b 52%, #111a22 100%);
  box-shadow: inset .8rem 0 1.8rem rgba(255,255,255,.16), inset -1rem 0 2rem rgba(0,0,0,.28), 0 1.8rem 3rem rgba(0,0,0,.44);
  transform-style: preserve-3d;
}

.ap-twin__lid {
  position: absolute;
  top: -.35rem;
  right: .7rem;
  left: .7rem;
  height: 1.1rem;
  border: 1px solid rgba(220,230,232,.28);
  border-radius: 1rem 1rem .35rem .35rem;
  background: linear-gradient(180deg, #c2ccce, #64747a);
  box-shadow: inset 0 1px rgba(255,255,255,.25), 0 .25rem .4rem rgba(0,0,0,.24);
  transform: translateZ(.4rem);
}

.ap-twin__lid:after {
  position: absolute;
  top: -.35rem;
  left: 50%;
  width: 2rem;
  height: .3rem;
  border-radius: .3rem;
  background: #27343a;
  box-shadow: inset 0 1px rgba(255,255,255,.18);
  content: '';
  transform: translateX(-50%);
}

.ap-twin__badge {
  position: absolute;
  top: 4.25rem;
  left: 1rem;
  color: rgba(232,241,241,.78);
  font-size: .45rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  transform: translateZ(1rem);
}

.ap-twin__vents {
  position: absolute;
  right: .75rem;
  bottom: 1rem;
  width: 2rem;
  height: 1.4rem;
  opacity: .7;
  background: repeating-linear-gradient(0deg, rgba(5,11,15,.72) 0 .12rem, transparent .12rem .34rem);
  transform: translateZ(.9rem);
}

.ap-twin__display {
  position: absolute;
  top: 1.65rem;
  left: 1.25rem;
  width: 3.2rem;
  height: 1rem;
  border: 1px solid rgba(190,224,218,.25);
  border-radius: .25rem;
  background: #15292b;
  box-shadow: inset 0 0 .4rem rgba(0,0,0,.5);
  transform: translateZ(1.35rem);
}

.ap-twin__display:after {
  position: absolute;
  top: .27rem;
  right: .35rem;
  left: .35rem;
  height: .2rem;
  border-radius: .2rem;
  background: linear-gradient(90deg, #8cc9bd 0 62%, #405956 62%);
  content: '';
}

.ap-twin__cartridge {
  position: absolute;
  bottom: 1.35rem;
  width: 2.15rem;
  height: 1.05rem;
  border: 1px solid rgba(210,223,227,.22);
  border-radius: .25rem;
  background: linear-gradient(90deg, #829198, #35434a);
  box-shadow: inset 0 1px rgba(255,255,255,.18);
  transform: translateZ(1.15rem);
}

.ap-twin__cartridge--one { left: 1.35rem; }
.ap-twin__cartridge--two { left: 4.1rem; }
.ap-twin__cartridge--three { right: 1.35rem; }

.ap-twin__foot {
  position: absolute;
  bottom: -.2rem;
  width: 1.4rem;
  height: .45rem;
  border-radius: 0 0 .3rem .3rem;
  background: #111a20;
  transform: translateZ(-.4rem);
}

.ap-twin__foot--left { left: 1.1rem; }
.ap-twin__foot--right { right: 1.1rem; }

.ap-tube {
  position: absolute;
  z-index: 2;
  width: 4.4rem;
  height: 2.4rem;
  border: .22rem solid rgba(130,190,183,.55);
  border-left-color: transparent;
  border-bottom-color: transparent;
  border-radius: 0 1.3rem 0 0;
  opacity: .8;
}

.ap-tube--inlet { top: 8.1rem; left: 36%; transform: rotate(12deg); }
.ap-tube--outlet { top: 15.2rem; right: 34%; transform: rotate(188deg); }

.ap-twin__panel {
  position: absolute;
  top: 1.2rem;
  right: 1rem;
  left: 1rem;
  height: 2.8rem;
  border: 1px solid rgba(210,223,227,.30);
  border-radius: .75rem;
  background: linear-gradient(120deg, rgba(218,229,232,.28), rgba(31,45,53,.38));
  box-shadow: inset 0 1px rgba(255,255,255,.16);
  transform: translateZ(1.2rem);
}

.ap-twin__panel:after {
  position: absolute;
  top: 1rem;
  right: 1rem;
  width: 1.2rem;
  height: .35rem;
  border-radius: 1rem;
  background: #7ec5b5;
  box-shadow: 0 0 .45rem rgba(126,197,181,.55);
  content: '';
}

.ap-twin__stage {
  position: absolute;
  right: 1.35rem;
  left: 1.35rem;
  height: 2.2rem;
  border: 1px solid rgba(221,231,233,.20);
  border-radius: .6rem;
  background: linear-gradient(90deg, rgba(8,15,21,.72), rgba(42,57,63,.58));
  transform: translateZ(.7rem);
}

.ap-twin__stage--one { top: 5.4rem; }
.ap-twin__stage--two { top: 8.3rem; }
.ap-twin__stage--three { top: 11.2rem; }

.ap-flow {
  position: absolute;
  top: 3.8rem;
  left: 50%;
  z-index: 3;
  width: .18rem;
  height: 13.2rem;
  transform: translateX(-50%);
  background: linear-gradient(to bottom, transparent, #a8d7ce, #6ea69e, transparent);
  box-shadow: 0 0 .55rem rgba(126,197,181,.45);
  opacity: .72;
}

.ap-flow:after {
  position: absolute;
  width: .65rem;
  height: .65rem;
  border-radius: 50%;
  background: #d7eee8;
  box-shadow: 0 0 .7rem rgba(126,197,181,.62);
  content: '';
  animation: ap-flow 2.2s linear infinite;
}

.ap-node {
  position: absolute;
  z-index: 5;
  min-width: 9.5rem;
  padding: .6rem .75rem;
  border: 1px solid rgba(195,211,216,.20);
  border-radius: .75rem;
  background: rgba(18,27,34,.92);
  color: #dce8f7;
  box-shadow: 0 .7rem 1.6rem rgba(0,0,0,.24);
  font-size: .7rem;
  transition: transform .3s cubic-bezier(.2,.8,.2,1), border-color .3s ease, box-shadow .3s ease;
}

.ap-node:hover {
  border-color: #8ab9b1;
  box-shadow: 0 0 1.2rem rgba(126,197,181,.14);
  transform: translateY(-.3rem) scale(1.03);
}

.ap-node:after {
  position: absolute;
  top: 50%;
  width: 3.4rem;
  border-top: 1px dashed rgba(151,187,184,.48);
  content: '';
}

.ap-node strong { display: block; color: #f3f7ff; font-size: .76rem; }
.ap-node span { color: #a7cfc7; }
.ap-node--left { top: 8rem; left: 1.1rem; }
.ap-node--left:after { left: 100%; }
.ap-node--right { top: 13rem; right: 1.1rem; }
.ap-node--right:after { right: 100%; }
.ap-node--top { top: 4.7rem; right: 1.1rem; }
.ap-node--top:after { right: 100%; }

.ap-gauge {
  position: absolute;
  right: 1.15rem;
  bottom: 1rem;
  z-index: 4;
  width: 5.2rem;
  height: 5.2rem;
}

.ap-gauge circle { fill: none; stroke-width: 5; }
.ap-gauge__track { stroke: rgba(255,255,255,.10); }
.ap-gauge__value { stroke: #83bdb2; stroke-linecap: round; filter: drop-shadow(0 0 4px rgba(126,197,181,.45)); stroke-dasharray: 125 157; transform: rotate(-90deg); transform-origin: 50% 50%; }
.ap-gauge text { fill: #effcff; font-family: 'Space Grotesk', sans-serif; font-size: .8rem; font-weight: 700; text-anchor: middle; }
.ap-gauge label { position: absolute; right: -.2rem; bottom: -.9rem; color: #8d9db3; font-size: .6rem; white-space: nowrap; }

@keyframes ap-flow { from { transform: translate(-.24rem, 0); } to { transform: translate(-.24rem, 12.6rem); } }
@keyframes ap-float { 0%, 100% { margin-top: 0; } 50% { margin-top: -.5rem; } }

@media (max-width: 900px) {
  .ap-bento { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ap-tile--hero, .ap-tile--wide { grid-column: span 2; }
}

@media (max-width: 800px) {
  [data-testid="stMetric"] { padding: .75rem; }
  .ap-hero { padding: 1.25rem; border-radius: 18px; }
  .ap-spatial { min-height: 31rem; }
  .ap-node { min-width: 7.5rem; font-size: .62rem; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service() -> AquaService:
    home = os.environ.get("AQUAPREDICT_HOME", "aquapredict_app_data")
    os.makedirs(home, exist_ok=True)
    return AquaService(os.environ.get("AQUAPREDICT_DB", os.path.join(home, "app.db")))


svc = get_service()
ERRORS = (NotFound, BadRequest, Conflict, ModelUnavailable)


# ------------------------------------------------------------------ helpers
def flash(kind, msg):
    st.session_state["_flash"] = (kind, msg)


def act(fn, ok_msg, *args, **kw):
    """Run a service action from a button callback; success/error shown at the top of the next run."""
    try:
        res = fn(*args, **kw)
        flash("success", ok_msg(res) if callable(ok_msg) else ok_msg)
        return res
    except ERRORS as e:
        flash("error", str(e.args[0]) if isinstance(e, KeyError) else str(e))
    except Exception as e:                       # never swallow silently: show the real error
        flash("error", f"{type(e).__name__}: {e}")


def goto(page, stage=None):
    st.session_state["page"] = page
    if stage:
        st.session_state["stage"] = stage


def fmt_window(fc):
    if fc["status"] == "withheld":
        return "Forecast withheld"
    p, iv = fc["point_days"], fc["interval_days"]
    if (iv[0] if iv else p) > 730:
        return "More than 2 years away (estimate very uncertain)"
    if iv:
        return f"Replace in {iv[0]:.0f}-{iv[1]:.0f} days (point estimate {p:.0f})"
    return f"About {p:.0f} days (point estimate, no calibrated interval)"


def pct(x):
    return "n/a" if x is None else f"{min(x, 1.5) * 100:.0f}%"


def spatial_twin(D, latest, focus):
    """Render a CSS 3D digital twin using live service values, without a new frontend build."""
    stage_progress = {}
    for stage in STAGES:
        condition = D["stages"][stage]["condition"]
        stage_progress[stage] = (
            f"{min(max(float(condition['progress']), 0.0), 1.0) * 100:.0f}%"
            if condition["available"] else "n/a"
        )
    ro_state = "Attention required" if D["stages"]["ro"]["sensor_health"] != "ok" else "Nominal"
    focus_label = STAGE_LABELS.get(focus, focus.title())
    return (
        '<section class="ap-spatial">'
        '<div class="ap-spatial__grid"></div>'
        f'<div class="ap-spatial__title">DIGITAL TWIN / FOCUS: {focus_label.upper()}</div>'
        '<div class="ap-twin">'
        '<div class="ap-twin__body"><div class="ap-twin__lid"></div>'
        '<div class="ap-twin__panel"></div><div class="ap-twin__display"></div>'
        '<div class="ap-twin__badge">AQUAPREDICT / AP-01</div>'
        '<div class="ap-twin__stage ap-twin__stage--one"></div>'
        '<div class="ap-twin__stage ap-twin__stage--two"></div>'
        '<div class="ap-twin__stage ap-twin__stage--three"></div>'
        '<div class="ap-twin__cartridge ap-twin__cartridge--one"></div>'
        '<div class="ap-twin__cartridge ap-twin__cartridge--two"></div>'
        '<div class="ap-twin__cartridge ap-twin__cartridge--three"></div>'
        '<div class="ap-twin__vents"></div><div class="ap-twin__foot ap-twin__foot--left"></div>'
        '<div class="ap-twin__foot ap-twin__foot--right"></div></div>'
        '<div class="ap-tube ap-tube--inlet"></div><div class="ap-tube ap-tube--outlet"></div></div>'
        '<div class="ap-flow"></div></div>'
        f'<div class="ap-node ap-node--left"><strong>Sediment stage</strong>'
        f'<span>{stage_progress["sediment"]} · pressure flow</span></div>'
        f'<div class="ap-node ap-node--top"><strong>RO membrane</strong>'
        f'<span>{ro_state}</span></div>'
        f'<div class="ap-node ap-node--right"><strong>Carbon stage</strong>'
        f'<span>{stage_progress["carbon"]} · usage load</span></div>'
        '<svg class="ap-gauge" viewBox="0 0 64 64" aria-label="System pressure gauge">'
        '<circle class="ap-gauge__track" cx="32" cy="32" r="25"></circle>'
        '<circle class="ap-gauge__value" cx="32" cy="32" r="25"></circle>'
        '<text x="32" y="35">LIVE</text><label>flow signal</label></svg>'
        '</section>'
    )


def day_to_date(hid, day):
    h = svc.household(hid)
    r = svc.readings(hid, last=1)
    d0 = pd.Timestamp(r["date"].iloc[-1]) - pd.Timedelta(days=int(r["day"].iloc[-1]))
    return (d0 + pd.Timedelta(days=int(day))).date().isoformat()


def sync_value(source, target):
    st.session_state[target] = st.session_state[source]


# ------------------------------------------------------------------ command center
households = svc.list_households()
models = [m for m in svc.models() if m["available"]]
model_names = [m["name"] for m in models]
labels = {m["name"]: m["label"] for m in svc.models()}
st.session_state.setdefault("page", PAGES[0] if households else "Simulator & what-if")
st.session_state.setdefault("top_page", st.session_state["page"])

if households:
    ids = [h["id"] for h in households]
    if st.session_state.get("hid") not in ids:
        st.session_state["hid"] = ids[-1]
    if "model" not in st.session_state or st.session_state["model"] not in model_names:
        st.session_state["model"] = svc.registry.default_model()
    st.session_state.setdefault("top_model", st.session_state["model"])
    hid = st.session_state.get("hid")
    model = st.session_state["model"]
    primary_pages = ["Dashboard", "Filter detail", "Simulator & what-if", "Alerts"]
    if st.session_state["top_page"] not in primary_pages:
        st.session_state["top_page"] = "Dashboard"
    brand, nav, selectors = st.columns([2.2, 4.6, 2.4], vertical_alignment="center")
    with brand:
        st.markdown(
            '<div class="ap-commandbar__brand"><i>💧</i><div>AquaPredict '
            '<span>water intelligence</span></div></div>',
            unsafe_allow_html=True,
        )
    with nav:
        st.markdown('<div class="ap-topnav">', unsafe_allow_html=True)
        st.radio("Page", primary_pages, key="top_page", horizontal=True, label_visibility="collapsed",
                 on_change=sync_value, args=("top_page", "page"))
        st.markdown('</div>', unsafe_allow_html=True)
    with selectors:
        household_col, gear_col = st.columns([4.3, 1], vertical_alignment="center")
        with household_col:
            st.selectbox("Purifier / household", ids, key="hid",
                         format_func=lambda i: f"{i} · {next(h['name'] for h in households if h['id'] == i)}")
        with gear_col:
            with st.popover("⚙️", use_container_width=False):
                st.caption("Simulator controls")
                st.selectbox("Forecast model", model_names, key="top_model", format_func=lambda n: labels[n],
                             on_change=sync_value, args=("top_model", "model"))
                if hid:
                    H = svc.household(hid)
                    if H["source"] == "synthetic":
                        st.number_input("Advance (days)", 1, 365, 14, key="top_adv_days",
                                        label_visibility="visible", on_change=sync_value,
                                        args=("top_adv_days", "adv_days"))
                        st.button("Advance clock", key="adv_btn_top", on_click=lambda: act(
                            svc.advance, lambda r: f"Advanced to day {r['now_day']}; forecasts, alerts and plan updated.", hid,
                            st.session_state["adv_days"], 7, model))
                    st.button("Refresh data", key="monitor_btn_top",
                              on_click=lambda: act(svc.update_monitoring, "Monitoring refreshed.", hid, model))
else:
    st.session_state.setdefault("page", "Simulator & what-if")
if "model" not in st.session_state or st.session_state["model"] not in model_names:
    st.session_state["model"] = svc.registry.default_model()
st.session_state.setdefault("top_model", st.session_state["model"])
hid = st.session_state.get("hid")
model = st.session_state["model"]
page = st.session_state["page"]

if hid:
    H = svc.household(hid)

# Retain hidden widget keys for scripted clients and existing automation. The visual
# command center above is the production surface; this compatibility surface is not rendered.
st.sidebar.radio("Page", PAGES, key="page", label_visibility="collapsed")
if households:
    st.sidebar.selectbox("Forecast model", model_names, key="model", label_visibility="collapsed",
                         format_func=lambda n: labels[n])
if hid and H["source"] == "synthetic":
    st.sidebar.number_input("Advance by (days)", 1, 365, 14, key="adv_days", label_visibility="collapsed")
    st.sidebar.button("Advance demo clock", key="adv_btn", on_click=lambda: act(
        svc.advance, lambda r: f"Advanced to day {r['now_day']}; forecasts, alerts and plan updated.", hid,
        st.session_state["adv_days"], 7, model))
if hid:
    st.sidebar.button("Refresh forecasts, alerts and plan", key="monitor_btn", on_click=lambda: act(
        svc.update_monitoring, "Monitoring refreshed.", hid, model))
    if H["source"] == "uploaded":
        # Kept in the hidden compatibility surface for AppTest and assistive
        # clients; the production UI presents this state in the quiet footer.
        st.sidebar.info("UPLOADED telemetry · connected to local SQLite")


# ------------------------------------------------------------------ pages
def page_dashboard():
    D = svc.dashboard(hid, model)
    st.markdown(
        f'<section class="ap-hero"><div class="ap-kicker">Live purifier workspace</div>'
        f'<h1>Dashboard <span style="opacity:.62;font-weight:500;">/ {D["household"]["name"]}</span></h1>'
        f'<p>{D["data_source_label"]} · Household <strong>{hid}</strong> · '
        f'Demo day {D["household"]["now_day"]}</p>'
        f'<div style="margin-top:1.1rem;"><span class="ap-ai-badge">MODEL READOUT · {labels[model]}</span></div>'
        f'</section>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ap-section-label">Workspace pulse</div>', unsafe_allow_html=True)
    latest = svc.readings(hid, last=1).iloc[0]
    def tile_value(column, digits=0):
        value = latest.get(column)
        if value is None or pd.isna(value):
            return "--"
        return f"{float(value):.{digits}f}"
    progress = [
        float(D["stages"][s]["condition"]["progress"])
        for s in STAGES if D["stages"][s]["condition"]["available"]
    ]
    wear = f"{max(progress) * 100:.0f}%" if progress else "--"
    st.markdown(
        f'<div class="ap-bento">'
        f'<article class="ap-tile ap-tile--hero">'
        f'<div class="ap-pulse">SYSTEM ONLINE · TELEMETRY LIVE</div>'
        f'<div class="ap-tile__value">{tile_value("flow_Lpm", 2)} '
        f'<span class="ap-tile__unit">L/min permeate flow</span></div>'
        f'<div class="ap-tile__meta">Active purifier · last reading day {int(latest["day"])} · '
        f'{D["data_source_label"]}</div><div class="ap-progress"><span></span></div>'
        f'<div class="ap-tile__meta">Estimated filter wear signal · {wear}</div></article>'
        f'<article class="ap-tile"><div class="ap-tile__label">Output TDS</div>'
        f'<div class="ap-tile__value">{tile_value("out_tds_ppm")} <span class="ap-tile__unit">ppm</span></div>'
        f'<div class="ap-tile__meta">Clean water quality signal</div></article>'
        f'<article class="ap-tile"><div class="ap-tile__label">Daily consumption</div>'
        f'<div class="ap-tile__value">{tile_value("usage_L")} <span class="ap-tile__unit">L/day</span></div>'
        f'<div class="ap-tile__meta">Current usage pattern</div></article>'
        f'<article class="ap-tile ap-tile--wide"><div class="ap-tile__label">System overview</div>'
        f'<div class="ap-tile__value">{len(households)} <span class="ap-tile__unit">purifiers</span>'
        f' · {D["n_filter_stages"]} <span class="ap-tile__unit">stages</span></div>'
        f'<div class="ap-tile__meta">{D["open_alerts"]} open alerts · {len(D["withheld"])} forecasts withheld · '
        f'{len(D["approaching"])} maintenance windows approaching</div></article>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ap-section-label">Spatial system view</div>', unsafe_allow_html=True)
    focus = st.radio(
        "Digital twin focus",
        list(STAGES),
        key="twin_focus",
        horizontal=True,
        format_func=lambda stage: STAGE_LABELS[stage],
        label_visibility="collapsed",
    )
    st.markdown(spatial_twin(D, latest, focus), unsafe_allow_html=True)
    dq = D["data_quality"]
    stale = {k: v for k, v in dq["days_since_valid_reading"].items() if v and v > 0}
    st.caption(f"Latest reading: day {dq['last_day']}" + (" · idle day: no water drawn" if dq["idle_today"] else "")
               + (f"  ·  stale sensors (days since valid reading): {stale}" if stale else "  ·  all sensors reporting"))
    st.markdown(
        '<div class="ap-note"><strong>Read the dashboard in three layers:</strong> '
        'filter <em>condition</em> (wear), <em>sensor health</em> (trust), and '
        '<em>forecast reliability</em> (confidence in timing).</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ap-section-label">Filter intelligence</div>', unsafe_allow_html=True)
    for s in STAGES:
        v = D["stages"][s]
        with st.container(border=True):
            a, b = st.columns([2, 3])
            a.subheader(v["label"])
            cond = v["condition"]
            a.markdown(f"**Condition:** {pct(cond['progress'])} of the way to the assumed criterion" if cond["available"]
                       else "**Condition:** not enough data yet")
            a.caption(cond.get("text", ""))
            a.markdown(f"**Sensor health:** {BADGE[v['sensor_health']]}")
            a.markdown(f"**Forecast reliability:** {v['reliability']}")
            b.markdown(f"### {fmt_window(v)}")
            if v["reason"]:
                b.warning(f"Why withheld: {v['reason']}")
            b.write(v["recommendation"])
            b.button("Open details", key=f"open_{s}", on_click=goto, args=("Filter detail", s))
    st.markdown('<div class="ap-section-label">Action queue</div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, .85])
    with left:
        st.subheader("Recent open alerts")
        for al in D["recent_alerts"] or []:
            st.markdown(f"- **{al['severity'].upper()}** · {al['title']}")
        if not D["recent_alerts"]:
            st.caption("No open alerts.")
        st.button("Open alert centre", key="goto_alerts", on_click=goto, args=("Alerts",))
    with right:
        st.subheader("Upcoming maintenance")
        for p in D["upcoming_maintenance"]:
            w = f"days {p['window_lo']}-{p['window_hi']}" if p["window_lo"] is not None else ""
            st.markdown(f"- {STAGE_LABELS[p['stage']]}: **{p['status']}** ({w})")
        if not D["upcoming_maintenance"]:
            st.caption("Nothing planned yet. Use 'Refresh forecasts, alerts and plan'.")
        if D["recent_maintenance"]:
            st.caption("Recently completed: " + ", ".join(f"{STAGE_LABELS[p['stage']]} (day {p['completed_day']})" for p in D["recent_maintenance"]))
    M = D["model"]
    with st.expander(f"Model and evaluation status: {labels[M['name']]}"):
        st.write(M["trained_with"] or "No training record (baseline or untrained).")
        if M["card"]:
            st.markdown(f"**What it is:** {M['card']['what']}")
            st.markdown(f"**Reported measured results (synthetic benchmark):** {M['card']['measured']}")
            st.markdown(f"**Limitations:** {M['card']['limits']}")
        st.caption("Reported numbers are not live. The Validation lab recomputes them on fresh households.")
    st.caption(D["disclaimer"])


def indicator_series(hist, stage):
    from aquapredict.baselines.baselines import _cycle
    cyc = _cycle(hist, stage)
    d = cyc["day"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if stage == "sediment":
            y = cyc["dp_sediment_bar"].to_numpy(float) / cyc["flow_Lpm"].to_numpy(float)
            return d, y / np.nanmedian(y[:7]), "pressure drop per flow (x clean)", 1.8
        if stage == "ro":
            T = cyc["temp_C"].to_numpy(float)
            y = cyc["flow_Lpm"].to_numpy(float) / (1 + 0.03 * (T - 25))
            return d, y / np.nanmedian(y[:7]), "temperature-corrected flow (x clean)", 0.70
    return d, cyc["cum_vol_carbon_L"].to_numpy(float), "litres since installation (usage-based)", None


def page_filter_detail():
    st.header("Filter detail")
    stage = st.radio("Filter stage", list(STAGES), key="stage", horizontal=True, format_func=lambda s: STAGE_LABELS[s])
    try:
        fc = svc.forecast(hid, stage, model, explain=True)
    except ERRORS as e:
        st.error(str(e)); return
    st.markdown(f"### {fmt_window(fc)}")
    c = st.columns(3)
    c[0].metric("Forecast status", fc["status"].replace("_", " "))
    c[1].metric("Sensor health", BADGE[fc["sensor_health"]])
    c[2].metric("Model", labels[fc["model"]])
    st.write(f"**Reliability:** {fc['reliability']}")
    st.caption(f"Interval basis: {fc['interval_status'] or 'n/a'}  ·  model version: {fc['model_version']}")
    if fc["reason"]:
        st.warning(f"Forecast withheld or limited: {fc['reason']}")
    st.info(f"**Recommended next action:** {fc['recommendation']}")
    st.markdown(f"**Filter condition:** {fc['indicator']['text'] or 'not enough data'}")
    t1, t2, t3, t4 = st.tabs(["Why this forecast", "Sensor and data quality", "Trends", "Simulator truth (demo)"])
    with t1:
        if fc["drivers"]:
            st.dataframe(pd.DataFrame(fc["drivers"]), use_container_width=True)
        if fc["explanation"] and fc["explanation"].get("available"):
            st.write(fc["explanation"].get("summary", ""))
        if not fc["drivers"]:
            st.caption("This model does not provide per-forecast drivers. Select 'Person 2 model' (trained) for drivers; "
                       "the filter-condition line above is the sensor-based evidence for this stage.")
        sc = fc.get("source_change")
        if sc and sc.get("detected"):
            st.warning(f"Possible water-source or operating-condition change: {sc.get('message', '')} (statistical signal only)")
    with t2:
        evs = fc["sensor_events"] + [e for e in fc["fault_events"]]
        if evs:
            st.dataframe(pd.DataFrame(evs), use_container_width=True)
            st.caption("All events are SUSPECTED, not confirmed. A suspected fault is not proof of a broken sensor.")
        else:
            st.success("No sensor or data-quality events for this stage.")
        st.json(svc.data_quality(hid)["days_since_valid_reading"], expanded=False)
    with t3:
        hist = svc.history(hid)
        d, y, label, crit = indicator_series(hist, stage)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
        ax[0].plot(d, y, ".", ms=3); ax[0].set_title(label, fontsize=9); ax[0].set_xlabel("day")
        if crit:
            ax[0].axhline(crit, ls="--", c="r", lw=.8)
        ph = svc.prediction_history(hid, stage, model)
        ph = ph[ph["status"] != "withheld"] if len(ph) else ph
        if len(ph):
            ax[1].plot(ph["day"], ph["day"] + ph["point"], "o-", ms=3, label="predicted replacement day")
            if ph["lo"].notna().any():
                ax[1].fill_between(ph["day"], ph["day"] + ph["lo"], ph["day"] + ph["hi"], alpha=.2, label="interval")
            ax[1].legend(fontsize=7)
        ax[1].set_title("predicted replacement day over time", fontsize=9); ax[1].set_xlabel("forecast made on day")
        st.pyplot(fig); plt.close(fig)
        st.button("Backfill forecast history for this stage (past days only)", key="backfill", on_click=lambda: act(
            backfill, lambda n: f"Computed {n} past forecasts.", stage))
    with t4:
        T = svc.truth(hid, stage)
        if not T["available"]:
            st.info(T["note"])
        elif T.get("censored"):
            st.write(T["note"])
        else:
            st.warning("DEMO ONLY: this is the simulator's hidden ground truth. It does not exist for real purifiers.")
            st.metric("True days until the criterion is reached", f"{T['remaining_days_true']:.0f}")
            if fc["point_days"] is not None:
                st.write(f"Forecast point estimate: {fc['point_days']:.0f} days (error {fc['point_days'] - T['remaining_days_true']:+.0f} d).")


def backfill(stage):
    H = svc.household(hid)
    hist = svc.history(hid)
    start = int(hist["day"].iloc[-1]) - int(hist[f"age_{stage}_d"].iloc[-1]) + 14
    days = list(range(max(start, 14), H["now_day"], 7))
    for d in days:
        svc.forecast(hid, stage, model, day=d, store=True, explain=False)
    return len(days)


def parse_json(text, what):
    try:
        v = json.loads(text or "{}")
    except json.JSONDecodeError as e:
        raise BadRequest(f"{what} is not valid JSON: {e}")
    if not isinstance(v, dict):
        raise BadRequest(f"{what} must be a JSON object")
    return v


def _create():
    ss = st.session_state
    h = svc.create_household(ss["new_name"], ss["new_scenario"], ss["new_law"], int(ss["new_seed"]), None,
                             parse_json(ss["new_overrides"], "Overrides"), ss["new_calibrated"], int(ss["new_now"]))
    ss["hid"] = h["id"]
    ss["page"] = "Dashboard"
    ss["top_page"] = "Dashboard"
    svc.update_monitoring(h["id"], ss.get("model"))
    return h


def _fault_spec(prefix):
    ss = st.session_state
    return {"kind": ss[f"{prefix}_kind"], "sensor": ss[f"{prefix}_sensor"], "start": int(ss[f"{prefix}_start"]),
            "duration": int(ss[f"{prefix}_dur"]), "magnitude": float(ss[f"{prefix}_mag"])}


def fault_widgets(prefix, default_start):
    c = st.columns(5)
    c[0].selectbox("Fault type", ["stuck", "drift", "spike", "dropout", "tds_inversion"], key=f"{prefix}_kind")
    c[1].selectbox("Sensor", SENSOR_OPTIONS, key=f"{prefix}_sensor")
    c[2].number_input("Starts on day", 0, 2000, int(default_start), key=f"{prefix}_start")
    c[3].number_input("Duration (days)", 1, 400, 20, key=f"{prefix}_dur")
    c[4].number_input("Magnitude", -5.0, 50.0, 0.3, key=f"{prefix}_mag",
                      help="drift: signed fraction (0.3 = +30%); spike / TDS inversion: multiplier")


def page_simulator():
    st.header("Digital purifier simulator and what-if")
    st.caption("A reproducible simulator generates the telemetry. It is a model with illustrative parameters, not a validated physical "
               "twin. Everything shown is generated by the real simulator code.")
    with st.expander("Create a synthetic household", expanded=not households):
        scs = svc.scenarios()
        c = st.columns(4)
        c[0].text_input("Name", "My demo home", key="new_name")
        c[1].selectbox("Scenario", [s["name"] for s in scs], key="new_scenario",
                       format_func=lambda n: f"{n} - {next(s['description'] for s in scs if s['name'] == n)[:60]}")
        c[2].selectbox("Wear laws", ["A", "B"], key="new_law", help="B = second simulator with different wear equations")
        c[3].number_input("Random seed", 0, 10**6, 7, key="new_seed")
        c = st.columns(3)
        c[0].number_input("Start the demo on day", 0, 999, 60, key="new_now")
        c[1].checkbox("Use the calibrated simulator (recommended)", True, key="new_calibrated")
        c[2].text_area("Parameter overrides (JSON, optional)", "{}", key="new_overrides", height=70,
                       help="e.g. {\"usage_median_L\": 70, \"missing_rate\": 0.1}. See docs/SIMULATOR.md for all parameters.")
        st.button("Create household", key="create_btn", on_click=lambda: act(_create, lambda h: f"Created {h['id']} ({h['scenario']})."))
    if not hid:
        return
    H = svc.household(hid)
    if H["source"] != "synthetic":
        st.info("Simulator controls apply to synthetic households only."); return
    st.subheader(f"Controls for {hid}")
    st.markdown("**Record that you replaced a filter today** (re-runs the simulator: earlier readings are unchanged, the new filter follows)")
    cols = st.columns(3)
    for s, col in zip(STAGES, cols):
        col.button(f"Record {STAGE_LABELS[s]} replacement (day {H['now_day']})", key=f"rep_{s}",
                   on_click=lambda s=s: act(svc.record_replacement, f"{STAGE_LABELS[s]} replacement recorded.", hid, s))
    st.markdown("**Inject a sensor fault** (starts today or later; the past cannot be rewritten)")
    fault_widgets("inj", H["now_day"])
    st.button("Inject fault", key="inj_btn", on_click=lambda: act(
        lambda: svc.inject_fault(hid, _fault_spec("inj")), "Fault injected. Advance the demo clock to see its effect."))
    if H["user_faults"]:
        st.caption("Faults injected so far: " + "; ".join(f"{f['kind']} on {f.get('sensor')} from day {f['start']}" for f in H["user_faults"]))
    st.markdown("---")
    st.subheader("What-if (HYPOTHETICAL scenario)")
    st.caption("Re-simulates this household with changed parameters and compares forecasts at the same day. Nothing is saved, and it describes "
               "a changed synthetic world, not your actual purifier.")
    c = st.columns(2)
    c[0].radio("Stage", list(STAGES), key="w_stage", horizontal=True, format_func=lambda s: STAGE_LABELS[s])
    c[1].text_area("Parameter overrides (JSON)", "{\"usage_median_L\": 80}", key="w_over", height=70)
    st.checkbox("Also inject a hypothetical sensor fault", False, key="w_fault")
    if st.session_state.get("w_fault"):
        fault_widgets("wf", max(0, H["now_day"] - 30))
    def _whatif():
        ss = st.session_state
        ss["_whatif"] = svc.what_if(hid, parse_json(ss["w_over"], "Overrides"),
                                    [_fault_spec("wf")] if ss.get("w_fault") else [], model, ss["w_stage"])
    st.button("Run what-if", key="w_btn", on_click=lambda: act(_whatif, "What-if computed."))
    W = st.session_state.get("_whatif")
    if W:
        st.warning(W["label"])
        rows = [{"": "Current household", "status": W["base"]["status"], "point (days)": W["base"]["point_days"],
                 "interval": W["base"]["interval_days"], "simulator truth (days)": (W["simulator_truth_base"] or {}).get("remaining_days_true")},
                {"": "Hypothetical", "status": W["hypothetical"]["status"], "point (days)": W["hypothetical"]["point_days"],
                 "interval": W["hypothetical"]["interval_days"], "simulator truth (days)": W["simulator_truth_hypothetical"]}]
        st.dataframe(pd.DataFrame(rows).astype(str), use_container_width=True)
    st.markdown("---")
    st.subheader("Latest readings")
    st.dataframe(svc.readings(hid, last=20), use_container_width=True)


def page_alerts():
    st.header("Alert centre")
    st.caption("Model-generated warnings and your own actions are kept separate. Alerts are de-duplicated: the same condition updates one "
               "alert instead of creating a new one on every refresh.")
    c = st.columns(3)
    status = c[0].multiselect("Status", ["open", "acknowledged", "resolved", "dismissed"], ["open", "acknowledged"], key="al_status")
    stage = c[1].selectbox("Stage", ["all"] + list(STAGES), key="al_stage")
    c[2].button("Refresh monitoring now", key="al_refresh", on_click=lambda: act(svc.update_monitoring, "Monitoring refreshed.", hid, model))
    rows = [a for a in svc.alerts(hid) if a["status"] in status and (stage == "all" or a["stage"] == stage)]
    if not rows:
        st.success("No alerts match these filters.")
    for a in rows:
        icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}[a["severity"]]
        with st.expander(f"{icon} {a['title']}  ·  {a['status']}  ·  day {a['first_day']}-{a['last_day']}"):
            st.write(a["message"])
            st.markdown(f"**Recommended action:** {a['recommended_action']}")
            st.caption(f"origin: {a['origin']}  ·  kind: {a['kind']}  ·  notes: {a['note'] or '-'}")
            st.text_input("Note (optional)", key=f"note_{a['id']}")
            b = st.columns(3)
            for col, action in zip(b, ("acknowledge", "resolve", "dismiss")):
                col.button(action.capitalize(), key=f"{action}_{a['id']}", on_click=lambda i=a["id"], ac=action: act(
                    svc.alert_action, f"Alert {i}: {ac}d.".replace("ee", "e"), i, ac, st.session_state.get(f"note_{i}") or None))


def page_planner():
    st.header("Maintenance planner and costs")
    st.caption("Distinct states: **predicted** (forecast window) → **recommended** (window is close) → **confirmed** (you chose a date) → "
               "**completed** (you recorded the replacement). The app never books a service or buys a cartridge.")
    rows = svc.plan(hid)
    if not rows:
        st.info("No plan entries yet. Use 'Refresh forecasts, alerts and plan' in the sidebar.")
    for p in rows:
        icon = {"predicted": "🔵", "recommended": "🟠", "confirmed": "🟣", "completed": "🟢", "cancelled": "⚪"}[p["status"]]
        win = f"days {p['window_lo']}-{p['window_hi']}" if p["window_lo"] is not None else "-"
        with st.container(border=True):
            st.markdown(f"{icon} **{STAGE_LABELS[p['stage']]}** · {p['status']} · window {win}"
                        + (f" · planned day {p['planned_day']}" if p["planned_day"] is not None else "")
                        + (f" · completed day {p['completed_day']}" if p["completed_day"] is not None else ""))
            st.caption(p["note"] or "")
            if p["status"] in ("predicted", "recommended", "confirmed"):
                c = st.columns(4)
                if p["status"] != "confirmed":
                    c[0].number_input("Planned day", 0, 2000, int(p["window_lo"] or svc.household(hid)["now_day"]), key=f"pd_{p['id']}")
                    c[1].button("Confirm date", key=f"cf_{p['id']}", on_click=lambda i=p["id"]: act(
                        svc.confirm_plan, "Date confirmed.", i, st.session_state[f"pd_{i}"]))
                c[2].button("Mark replaced today", key=f"cp_{p['id']}", on_click=lambda i=p["id"]: act(
                    svc.complete_plan, "Replacement recorded.", i))
                c[3].button("Cancel entry", key=f"cx_{p['id']}", on_click=lambda i=p["id"]: act(svc.cancel_plan, "Entry cancelled.", i))
    st.markdown("---")
    st.subheader("Cost estimate")
    st.warning("Prices are ILLUSTRATIVE placeholders you can edit (only the RO price comes from the abstract's reference and needs re-verifying). "
               "These are not manufacturer prices and not measured savings.")
    d = CostConfig()
    c = st.columns(5)
    prices = {s: c[i].number_input(f"{STAGE_LABELS[s]} (Rs)", 0.0, 100000.0, float(d.replacement_cost[s]), key=f"price_{s}") for i, s in enumerate(STAGES)}
    visit = c[3].number_input("Service visit (Rs)", 0.0, 100000.0, float(d.service_visit_cost), key="price_visit")
    horizon = c[4].number_input("Horizon (days)", 7, 730, 90, key="price_h")
    est = svc.cost_estimate(hid, int(horizon), CostConfig(replacement_cost=prices, service_visit_cost=visit))
    if est["lines"]:
        st.dataframe(pd.DataFrame(est["lines"]).astype(str), use_container_width=True)
    st.metric(f"Estimated spend in the next {horizon} days", f"Rs {est['total']:,.0f}", help="parts + one service visit")
    st.caption("Fixed-schedule vs forecast-based policy costs, with price sensitivity, are computed in the Validation lab on simulated households.")


def show_table(title, records, **kw):
    if records:
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame(records).round(3), use_container_width=True, **kw)


def page_validation():
    st.header("Validation lab")
    st.caption("Scores forecasters on FRESH synthetic households with the project's strict evaluation: only past readings are used, criteria are "
               "paired and use preset thresholds. Results are synthetic: they show algorithmic feasibility, not real-world accuracy.")
    avail = [m["name"] for m in svc.models() if m["available"]]
    c = st.columns(4)
    c[0].number_input("Households", 4, 60, 10, key="v_n")
    c[1].number_input("Random seed", 0, 10**6, 4242, key="v_seed", help="Must differ from a learned model's training seed")
    c[2].multiselect("Methods (baselines are always included)", [m for m in avail if m not in ("trend", "volume_rule", "health_20pct")],
                     key="v_methods", help="Person 2's model takes about 15 s per household")
    c[3].checkbox("Calibrated simulator", True, key="v_cal")
    def _run():
        ss = st.session_state
        ss["_val"] = svc.run_validation(int(ss["v_n"]), int(ss["v_seed"]), ["trend", "volume_rule", "health_20pct"] + list(ss["v_methods"]),
                                        ss["v_cal"])
    st.button("Run validation", key="v_run", on_click=lambda: act(_run, lambda _: "Validation finished."))
    R = st.session_state.get("_val")
    if R:
        st.success(f"{R['n_forecasts']:,} forecasts on {R['params']['n_households']} fresh households, {R['n_filter_lives']} filter lives "
                   f"({R['n_short_excluded']} too short to forecast, excluded for every method). Took {R['seconds']} s.")
        show_table("Window error (days, forecasts within 60 d of the criterion)", R["window"])
        show_table("Late warnings and premature (false) alarms", R["late_false"])
        show_table("Prediction intervals (coverage target 90%)", R["intervals"])
        show_table("Window error by stage", R["by_stage"])
        for m, crit in R["criteria"].items():
            st.markdown(f"**Success criteria for {labels.get(m, m)}: " + ("ALL MET" if crit["all_passed"] else "NOT all met") + "**")
            for k, v in crit.items():
                if k != "all_passed":
                    st.write(("✅ " if v["passed"] else "❌ ") + k + ": " + v["rule"])
        show_table("Replacement-policy cost (ILLUSTRATIVE prices, simulated)", R["cost_policies"])
        with st.expander("Metric definitions"):
            st.code(R["definitions"])
        st.caption(R["caveat"])
        st.download_button("Download this result (JSON)", json.dumps(R, indent=2), "validation_result.json", "application/json", key="v_dl")
    past = svc.experiments()
    if past:
        st.markdown("---")
        st.caption(f"{len(past)} saved experiments (also in Data & export).")


def page_models():
    st.header("Models")
    st.caption("Learned models are trained on fresh synthetic households (calibrated simulator) with separate calibration households.")
    for m in svc.models():
        with st.expander(f"{'✅' if m['available'] else '⬜'} {m['label']}", expanded=False):
            if m["card"]:
                st.markdown(f"**What it is:** {m['card']['what']}")
                st.markdown(f"**Reported measured results:** {m['card']['measured']}")
                st.markdown(f"**Limitations:** {m['card']['limits']}")
            st.write(m["trained_with"] or m["reason"] or "baseline (no training)")
    st.markdown("---")
    st.subheader("Train models")
    st.warning("Takes a few minutes (about 2-4 for 120 households). The page waits until training finishes.")
    c = st.columns(4)
    c[0].multiselect("Models", ["reference_v3", "person2"], ["reference_v3", "person2"], key="t_which")
    c[1].number_input("Households", 24, 400, 120, key="t_n")
    c[2].number_input("Training seed", 0, 10**6, 1000, key="t_seed")
    c[3].number_input("Boosting iterations", 20, 600, 200, key="t_iter")
    st.button("Train", key="t_btn", on_click=lambda: act(
        lambda: svc.registry.train(tuple(st.session_state["t_which"]), int(st.session_state["t_n"]), int(st.session_state["t_seed"]),
                                   int(st.session_state["t_iter"]), log=lambda *_: None), "Training finished."))


def page_data():
    st.header("Data and export")
    st.subheader("Live public water-quality context")
    st.caption("This panel uses the USGS National Water Information System. It is environmental station data, not purifier telemetry, and it is never used as filter-life ground truth.")
    usgs_site = st.text_input("USGS station id", "01646500", key="usgs_site")
    if st.button("Fetch latest USGS observations", key="usgs_fetch"):
        try:
            st.session_state["usgs_data"] = svc.online_water_quality(usgs_site)
        except Exception as exc:
            st.error(f"Could not fetch the public source: {exc}")
    usgs = st.session_state.get("usgs_data")
    if usgs:
        st.success(f"Source verified: {usgs['source']} · {usgs['site_name']}")
        st.dataframe(pd.DataFrame(usgs["observations"]), use_container_width=True)
        st.caption(f"Retrieved from {usgs['api_url']} · provisional observations may be revised.")
        st.warning("These observations provide water-quality context only. Upload household purifier telemetry to enable purifier filter forecasting.")
    st.markdown("---")
    c = st.columns(2)
    with c[0]:
        st.subheader("Downloads")
        truth = st.checkbox("Include simulator ground-truth columns (demo only)", False, key="ex_truth")
        for kind, fn in (("telemetry", f"{hid}_telemetry.csv"), ("predictions", "predictions.csv"), ("alerts", "alerts.csv"),
                         ("maintenance", "maintenance.csv"), ("experiments", "experiments.csv")):
            st.download_button(f"{kind.capitalize()} CSV", svc.export_csv(kind, hid if kind != "experiments" else None, truth),
                               fn, "text/csv", key=f"dl_{kind}")
        st.download_button("Household configuration (JSON)", json.dumps(svc.export_household_config(hid), indent=2),
                           f"{hid}_config.json", "application/json", key="dl_cfg")
        st.caption("Exports carry the data_source column; synthetic data are never presented as real telemetry.")
    with c[1]:
        st.subheader("Upload real telemetry (integration boundary)")
        st.caption("CSV with one row per day: `day` (0,1,2,...), `usage_L`, and any of "
                   "`dp_sediment_bar, flow_Lpm, out_tds_ppm, feed_tds_ppm, temp_C, inlet_bar` (blank = missing). Filter ages are derived from the "
                   "replacement days you list. At least 15 days. Nothing is repaired silently; invalid files are rejected with reasons.")
        up = st.file_uploader("Telemetry CSV", type=["csv"], key="up_file")
        st.text_input("Household name", "Uploaded purifier", key="up_name")
        st.text_input("Replacement days (JSON, e.g. {\"ro\": [120]})", "{}", key="up_repl")
        def _ingest():
            ss = st.session_state
            if ss.get("up_file") is None:
                raise BadRequest("choose a CSV file first")
            df = pd.read_csv(ss["up_file"])
            rows = json.loads(df.to_json(orient="records"))
            h = svc.ingest_telemetry(ss["up_name"], rows, parse_json(ss["up_repl"], "Replacement days"))
            ss["hid"] = h["id"]
            svc.update_monitoring(h["id"], ss.get("model"))
            return h
        st.button("Ingest", key="up_btn", on_click=lambda: act(_ingest, lambda h: f"Ingested as {h['id']} (uploaded)."))


# ------------------------------------------------------------------ dispatch
if not households:
    st.title("💧 AquaPredict")
    st.info("No households yet. Create a synthetic demo household to begin.")
    page_simulator()
else:
    {"Dashboard": page_dashboard, "Filter detail": page_filter_detail, "Simulator & what-if": page_simulator, "Alerts": page_alerts,
     "Planner & costs": page_planner, "Validation lab": page_validation, "Models": page_models, "Data & export": page_data}[page]()

if "_flash" in st.session_state:
    kind, msg = st.session_state.pop("_flash")
    if kind == "error":
        st.error(msg)
    else:
        st.caption(f"✓ {msg}")

if hid:
    source_label = "synthetic demo telemetry" if H["source"] == "synthetic" else "uploaded telemetry"
    st.markdown(
        f'<footer class="ap-quiet-footer">AquaPredict · SQLite connected · {source_label} · '
        'Forecasts describe filter wear only and do not assess water safety.</footer>',
        unsafe_allow_html=True,
    )
