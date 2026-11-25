"""
app.py
------
IPCV Casting Inspection dashboard (Streamlit, light theme).

A short companion to the report. Five pages:

    Overview      dataset, imbalance set-up and the headline result
    Inspector     classify a test image or an upload with the deployed models
    Imbalance     how each model degrades as defects become rare
    Fixes         which imbalance-handling method works for which model
    Compare       head-to-head comparison, heat map and ranking

Start it from the project root:   streamlit run app.py
"""

import hashlib
import os
import pickle
import shutil
import zipfile
from pathlib import Path


def unpack_dataset_zips():
    """Online deployment: unzip dataset_zips/*.zip into data/casting_512x512/.

    The browser upload on GitHub cannot take 1,300 separate images, so the
    unchanged class folders are stored as two zip files and unpacked here once,
    before the package looks for the data folder.
    """
    root = Path(__file__).resolve().parent
    target = root / "data" / "casting_512x512"
    for z in sorted((root / "dataset_zips").glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            folder = zf.namelist()[0].split("/")[0]          # def_front or ok_front
            if (target / folder).is_dir():
                continue
            tmp = target / f".tmp_{folder}"
            shutil.rmtree(tmp, ignore_errors=True)
            zf.extractall(tmp)
            (tmp / folder).rename(target / folder)
            shutil.rmtree(tmp, ignore_errors=True)


unpack_dataset_zips()

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from casting_imbalance.config import CFG
from casting_imbalance.data import CastingDataset, DataSplitter
from casting_imbalance.evaluation import Evaluator
from casting_imbalance.inference import InspectionService

# ---------------------------------------------------------------- edit me
STUDENT_NAME = "Andly Danny Anafiah"
STUDENT_ID = "0137461"
COURSE = "XBDS3024N · Image Processing and Computer Vision"

# ================================================================ palette
C = dict(bg="#F4F6F9", card="#FFFFFF", text="#14212F", muted="#5B6B7C", border="#E0E6EE",
         accent="#D9731F", accent2="#1F5F8B", ok="#13995B", bad="#D63B3B", warn="#C27409", grey="#9AA6B2")

MODELS = ["SVM", "LogReg", "KNN", "RandomForest", "MobileNetV2", "CNN"]
MODEL_NAME = {"SVM": "SVM", "LogReg": "Logistic Regression", "KNN": "k-NN",
              "RandomForest": "Random Forest", "MobileNetV2": "MobileNetV2", "CNN": "CNN"}
MODEL_FAMILY = {"SVM": "Classical", "LogReg": "Classical", "KNN": "Classical",
                "RandomForest": "Classical", "MobileNetV2": "Neural", "CNN": "Neural"}
MODEL_COLOUR = {"SVM": "#0072B2", "LogReg": "#E69F00", "KNN": "#8C564B",
                "RandomForest": "#009E73", "MobileNetV2": "#D55E00", "CNN": "#CC79A7"}
METHODS = ["none", "class_weight", "random_oversampling", "smote", "random_undersampling",
           "augmentation_oversampling", "threshold_moving"]
METHOD_NAME = {"none": "No fix", "class_weight": "Class weighting",
               "random_oversampling": "Random oversampling", "smote": "SMOTE",
               "random_undersampling": "Random undersampling",
               "augmentation_oversampling": "Augmentation oversampling",
               "threshold_moving": "Threshold moving"}
METHOD_COLOUR = {"none": "#9AA6B2", "class_weight": "#0072B2", "random_oversampling": "#56B4E9",
                 "smote": "#009E73", "random_undersampling": "#E69F00",
                 "augmentation_oversampling": "#CC79A7", "threshold_moving": "#D55E00"}
RATIOS = ["natural", "3", "10", "20"]
RATIO_NAME = {"natural": "Natural (0.66:1)", "3": "3 : 1", "10": "10 : 1", "20": "20 : 1"}
RATIO_SHORT = {"natural": "Natural", "3": "3:1", "10": "10:1", "20": "20:1"}
METRIC_NAME = {"balanced_accuracy": "Balanced accuracy", "recall_defective": "Defect recall",
               "recall_ok": "OK recall", "f1_defective": "F1 (defective)", "mcc": "MCC",
               "roc_auc": "ROC-AUC", "accuracy": "Accuracy"}
N_TRAIN_OK, N_TRAIN_DEF = 363, 547

PAGES = [("overview", "🏠", "Overview"), ("inspect", "🔍", "Inspector"), ("imbalance", "📉", "Imbalance"),
         ("fixes", "🛠️", "Fixes"), ("compare", "⚖️", "Compare")]

CSS = f"""
<style>
.stApp {{ background: {C['bg']}; }}
.block-container {{ padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1200px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
section[data-testid="stSidebar"] > div:first-child {{
  background: linear-gradient(180deg, #0F2233 0%, #16324A 60%, #1B3B57 100%); }}
section[data-testid="stSidebar"] * {{ color: #E4ECF4; }}
section[data-testid="stSidebar"] .stButton {{ margin: -0.3rem 0; }}
section[data-testid="stSidebar"] .stButton button {{
  background: transparent; border: none; border-radius: 10px; padding: .5rem .8rem;
  justify-content: flex-start; min-height: 0; }}
section[data-testid="stSidebar"] .stButton button div {{ justify-content: flex-start; }}
section[data-testid="stSidebar"] .stButton button p {{ font-size: .98rem; text-align: left; }}
section[data-testid="stSidebar"] .stButton button:hover {{ background: rgba(255,255,255,.09); }}
section[data-testid="stSidebar"] .stButton button[kind="primary"] {{
  background: {C['accent']}; box-shadow: 0 4px 14px rgba(217,115,31,.35); }}
section[data-testid="stSidebar"] .stButton button[kind="primary"] p {{ font-weight: 700; color: #fff; }}
.brand {{ text-align:center; padding: .1rem 0 1rem; }}
.brand .n {{ font-size: 1.2rem; font-weight: 800; margin-top: .25rem; line-height: 1.2; }}
.brand .line {{ width: 46px; height: 3px; background: {C['accent']}; border-radius: 2px; margin: .5rem auto 0; }}
.side-card {{ background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.12);
  border-radius: 12px; padding: .65rem .8rem; margin: 1.2rem 0 .4rem; font-size: .8rem; }}
.side-card .h {{ font-weight: 700; font-size: .85rem; margin-bottom: .25rem; }}
.ptitle {{ font-size: 1.9rem; font-weight: 800; color:{C['text']}; line-height:1.2; margin:0 0 .2rem; }}
.psub {{ color:{C['muted']}; font-size:.98rem; margin-bottom: .9rem; }}
.sec {{ font-size: 1.12rem; font-weight: 750; color:{C['text']}; margin: 1.3rem 0 .4rem; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:{C['card']}; border-radius:14px !important;
  border-color:{C['border']} !important; box-shadow: 0 1px 3px rgba(0,0,0,.04); }}
.kpi {{ border:1px solid {C['border']}; border-radius:14px; padding:.75rem .95rem; height:100%; background:{C['card']}; }}
.kpi .h {{ color:{C['muted']}; font-weight:650; font-size:.8rem; }}
.kpi .v {{ font-size:1.6rem; font-weight:800; margin-top:.15rem; line-height:1.15; }}
.kpi .s {{ color:{C['muted']}; font-size:.76rem; margin-top:.15rem; }}
.hl {{ background: linear-gradient(135deg, #FFF6EE 0%, #FFFFFF 70%); border:1px solid #F3D5BB;
  border-left: 6px solid {C['accent']}; border-radius:14px; padding:.75rem 1.05rem; margin: .9rem 0;
  color:{C['text']}; font-size:.97rem; }}
.badge {{ display:inline-block; padding:.5rem 1.25rem; border-radius:999px; font-weight:800; font-size:1.2rem; color:#fff; }}
.badge.ok {{ background:{C['ok']}; }} .badge.def {{ background:{C['bad']}; }} .badge.review {{ background:{C['warn']}; }}
.pill {{ display:inline-block; padding:.12rem .55rem; border-radius:999px; font-weight:700; font-size:.78rem; }}
.pill.ok {{ color:{C['ok']}; background: rgba(19,153,91,.12); }}
.pill.def {{ color:{C['bad']}; background: rgba(214,59,59,.12); }}
.pill.review {{ color:{C['warn']}; background: rgba(194,116,9,.14); }}
.center {{ text-align:center; }}
.muted {{ color:{C['muted']}; font-size:.83rem; }}
</style>"""

LOGO = f"""<svg width="54" height="54" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
<rect x="2" y="2" width="60" height="60" rx="14" fill="rgba(255,255,255,.08)" stroke="{C['accent']}" stroke-width="2"/>
<circle cx="32" cy="32" r="19" fill="none" stroke="#DCE6F0" stroke-width="2.5"/><g fill="{C['accent']}">""" + "".join(
    f'<path transform="rotate({k * 60} 32 32)" d="M32 26 C36 20 41 18 45 19 C41 22 37 25 34 28 Z"/>' for k in range(6)
) + """</g><circle cx="32" cy="32" r="5.5" fill="#0F2233" stroke="#DCE6F0" stroke-width="2"/>
<path d="M44 44 L54 54" stroke="#DCE6F0" stroke-width="4" stroke-linecap="round"/></svg>"""


# ================================================================ helpers
def md(html: str, container=st) -> None:
    container.markdown(html, unsafe_allow_html=True)


def title(text: str, sub: str) -> None:
    md(f'<div class="ptitle">{text}</div><div class="psub">{sub}</div>')


def section(text: str) -> None:
    md(f'<div class="sec">{text}</div>')


def highlight(text: str) -> None:
    md(f'<div class="hl">{text}</div>')


def kpi(container, label: str, value: str, sub: str = "", colour: str = None) -> None:
    md(f'<div class="kpi"><div class="h">{label}</div><div class="v" style="color:{colour or C["text"]}">'
       f'{value}</div><div class="s">{sub}</div></div>', container)


def show_image(container, img, caption=None):
    container.image(img, caption=caption, width="stretch")


def show_table(df, container=st, **kw):
    container.dataframe(df, width="stretch", hide_index=True, **kw)


def show_chart(chart, container=st):
    chart = (chart.configure(background="transparent", font="sans-serif")
             .configure_axis(labelColor=C["muted"], titleColor=C["muted"], gridColor="#E8EDF3",
                             domainColor=C["border"], tickColor=C["border"], labelFontSize=12,
                             titleFontSize=12)
             .configure_legend(labelColor=C["text"], titleColor=C["muted"], labelFontSize=12)
             .configure_title(color=C["text"], anchor="start", fontSize=14)
             .configure_header(labelColor=C["text"], titleColor=C["text"], labelFontSize=13)
             .configure_view(stroke=None))
    container.altair_chart(chart, width="stretch")


def model_scale(models=MODELS):
    return alt.Scale(domain=[MODEL_NAME[m] for m in models], range=[MODEL_COLOUR[m] for m in models])


def model_axis():
    return alt.Axis(labelLimit=200, labelFontSize=13, labelColor=C["text"])


def css_class(verdict: str) -> str:
    return {"OK": "ok", "Defective": "def"}.get(verdict, "review")


def gauge_svg(score: float, thr: float, title: str) -> str:
    """Half-circle gauge: green zone below the threshold, red zone above it."""
    cx, cy, r = 110, 110, 88

    def pt(v, rad=r):
        a = np.pi * (1 - v)
        return cx + rad * np.cos(a), cy - rad * np.sin(a)

    def arc(v0, v1, colour):
        (x0, y0), (x1, y1) = pt(v0), pt(v1)
        return (f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 0 1 {x1:.1f},{y1:.1f}" fill="none" '
                f'stroke="{colour}" stroke-width="20"/>')
    nx, ny = pt(score, r - 4)
    tx0, ty0 = pt(thr, r - 14)
    tx1, ty1 = pt(thr, r + 14)
    colour = C["bad"] if score >= thr else C["ok"]
    ticks = "".join(f'<text x="{pt(v, r + 24)[0]:.0f}" y="{pt(v, r + 24)[1] + 4:.0f}" font-size="13" '
                    f'fill="{C["muted"]}" text-anchor="middle">{v:.1f}</text>' for v in (0, .5, 1))
    return f"""<div class="center"><svg viewBox="0 -6 220 156" width="100%" style="max-width:280px">
  {arc(0, thr, 'rgba(19,153,91,.28)')}{arc(thr, 1, 'rgba(214,59,59,.28)')}{ticks}
  <line x1="{tx0:.1f}" y1="{ty0:.1f}" x2="{tx1:.1f}" y2="{ty1:.1f}" stroke="{C['text']}" stroke-width="3"/>
  <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{colour}" stroke-width="4" stroke-linecap="round"/>
  <circle cx="{cx}" cy="{cy}" r="6" fill="{colour}"/>
  <text x="{cx}" y="{cy + 32}" font-size="30" font-weight="800" fill="{colour}" text-anchor="middle">{score:.3f}</text>
</svg><div class="muted" style="margin-top:-.3rem">{title} · threshold {thr:.2f}</div></div>"""


def confusion_chart(tn, fp, fn, tp, title="", height=220):
    cm = pd.DataFrame({"True": ["OK", "OK", "Defective", "Defective"],
                       "Predicted": ["OK", "Defective", "OK", "Defective"],
                       "Count": [tn, fp, fn, tp],
                       "Meaning": ["Correct accept", "False alarm", "Defect MISSED", "Defect caught"],
                       "Good": [1, 0, 0, 1]})
    base = alt.Chart(cm).encode(x=alt.X("Predicted:N", sort=["OK", "Defective"], axis=alt.Axis(labelAngle=0)),
                                y=alt.Y("True:N", sort=["OK", "Defective"]))
    heat = base.mark_rect(cornerRadius=8).encode(
        color=alt.Color("Good:N", scale=alt.Scale(domain=[1, 0], range=["#D7EFE2", "#F8DADA"]), legend=None),
        tooltip=["Meaning", "Count"])
    txt = base.mark_text(fontSize=22, fontWeight="bold", color=C["text"]).encode(text="Count:Q")
    lab = base.mark_text(fontSize=11, dy=20, color=C["muted"]).encode(text="Meaning:N")
    return (heat + txt + lab).properties(height=height, title=title)




# ============================================================== data layer
class StudyData:
    """Dataset split, experiment results and the stored test-set scores."""

    SOURCES = ("classical", "classical_extra", "mobilenet", "cnn")

    def __init__(self):
        self.split = self._split()
        tab = CFG.output_dir / "tables"
        self.summary = self._csv(tab / "summary_main.csv")
        self.stats = self._csv(tab / "statistics.csv")
        frames = [self._csv(tab / f"results_{n}.csv") for n in self.SOURCES]
        frames = [f for f in frames if f is not None]
        self.results = pd.concat(frames, ignore_index=True) if frames else None
        self.probs, self.y_test = self._probs()

    @property
    def main(self):
        return None if self.summary is None else self.summary[self.summary.tag == "main"].copy()

    def value(self, model, ratio, method, metric="balanced_accuracy"):
        d = self.main
        d = d[(d.model == model) & (d.ratio == ratio) & (d.method == method)]
        return float(d[metric].iloc[0]) if len(d) else float("nan")

    @staticmethod
    def _csv(path):
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if "ratio" in df:
            df["ratio"] = df["ratio"].astype(str)
        return df

    @staticmethod
    def _split():
        try:
            df = DataSplitter().load_or_create(CastingDataset().index)
            return df[df["path"].map(os.path.exists)]
        except FileNotFoundError:
            # Online deployment: the full dataset is not in the repository, so the
            # Inspector uses the unchanged sample test images in samples/.
            path = CFG.output_dir / "tables" / "data_split.csv"
            if not path.exists():
                return None
            df = pd.read_csv(path).assign(path="")
            samples = CFG.project_root / "samples"
            for sub in ("ok_front", "def_front"):
                for f in (samples / sub).glob("*.jp*g"):
                    df.loc[df["filename"] == f.name, "path"] = str(f)
            return df

    @classmethod
    def _probs(cls):
        probs, y = {}, None
        for n in cls.SOURCES:
            f = CFG.output_dir / f"probs_{n}.pkl"
            if f.exists():
                with open(f, "rb") as fh:
                    d = pickle.load(fh)
                probs.update(d["probs"])
                y = d["y_test"]
        return probs, y

    def test_images(self):
        if self.split is None:
            return pd.DataFrame(columns=["filename", "path", "label"])
        t = self.split[self.split["split"] == "test"]
        return t[t["path"].map(lambda p: bool(p) and os.path.exists(p))].reset_index(drop=True)

    def rep0_metrics(self, model, ratio, method):
        """Metrics and confusion counts of repetition 0 (for confusion matrices)."""
        key = ("main", model, ratio, 0, method)
        if key not in self.probs or self.results is None:
            return None
        r = self.results
        row = r[(r.tag == "main") & (r.model == model) & (r.ratio == ratio) & (r.rep == 0) & (r.method == method)]
        thr = float(row.threshold.iloc[0]) if len(row) else 0.5
        return Evaluator.compute(self.y_test, self.probs[key], thr)




# ================================================================== the app
class CastingDashboard:

    def __init__(self):
        st.set_page_config(page_title="IPCV Casting Inspection", page_icon="⚙️", layout="wide")
        md(CSS)
        st.session_state.setdefault("page", "overview")
        self.data = self._load_data()
        self.service = self._load_service()

    @staticmethod
    @st.cache_resource(show_spinner="Loading study results ...")
    def _load_data():
        return StudyData()

    @staticmethod
    @st.cache_resource(show_spinner="Loading the trained models ...")
    def _load_service():
        try:
            return InspectionService(CFG.model_dir)
        except Exception as err:            # app still works without the models
            return err

    def _models_ready(self) -> bool:
        if isinstance(self.service, Exception):
            st.error("Trained models not found. Run `python scripts/train_final_models.py` first.")
            return False
        return True

    def _results_ready(self) -> bool:
        if self.data.summary is None:
            st.warning("Result tables not found. Run the experiment scripts and "
                       "`python scripts/analyse_results.py` first.")
            return False
        return True

    # -------------------------------------------------------- navigation
    @staticmethod
    def go(page: str) -> None:
        st.session_state["page"] = page

    def sidebar(self) -> str:
        sb = st.sidebar
        md(f'<div class="brand">{LOGO}<div class="n">IPCV Casting<br>Inspection</div>'
           '<div class="line"></div></div>', sb)
        page = st.session_state["page"]
        for pid, icon, label in PAGES:
            sb.button(f"{icon}  {label}", key=f"nav_{pid}", width="stretch",
                      type="primary" if pid == page else "secondary", on_click=self.go, args=(pid,))
        md(f'<div class="side-card"><div class="h">🎓 {STUDENT_NAME}</div>{STUDENT_ID}<br>'
           f'<span style="opacity:.75">{COURSE}</span></div>', sb)
        return page

    def pick_test_image(self, key: str, only_label: int = None):
        tests = self.data.test_images()
        if tests.empty:
            st.info("Put the dataset in the `data` folder to use the test images, or upload a photo.")
            return None
        if tests["path"].str.contains("samples").all():
            st.caption(f"{len(tests)} sample test images from Dabhi (2020), CC BY-NC-ND 4.0.")
        names = tests["filename"].tolist()
        sk = f"{key}_choice"
        st.session_state.setdefault(sk, tests[tests.label == 1].filename.iloc[0])
        st.caption("🎲 Pick a random test image")
        b1, b2, b3 = st.columns(3)
        if b1.button("Any", key=f"{key}_r", width="stretch"):
            st.session_state[sk] = str(np.random.choice(names))
        if b2.button("OK", key=f"{key}_ro", width="stretch"):
            st.session_state[sk] = str(np.random.choice(tests[tests.label == 0].filename))
        if b3.button("Defect", key=f"{key}_rd", width="stretch"):
            st.session_state[sk] = str(np.random.choice(tests[tests.label == 1].filename))
        name = st.selectbox("Test image (never used in training)", names, key=sk)
        row = tests[tests.filename == name].iloc[0]
        return name, cv2.imread(row.path, cv2.IMREAD_COLOR), int(row.label)

    @staticmethod
    def read_upload(f):
        img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        return (f.name, img, None) if img is not None else None

    @staticmethod
    def unfamiliar(colour) -> list:
        c = colour.astype(np.int16)
        reasons = []
        if float(np.mean(np.abs(c[..., 0] - c[..., 1]) + np.abs(c[..., 1] - c[..., 2]))) > 6:
            reasons.append("it is a colour photo (training photos are grayscale)")
        h, w = colour.shape[:2]
        if max(h, w) / min(h, w) > 1.15:
            reasons.append(f"it is not square ({w}×{h} px)")
        return reasons

    def scores(self, gray) -> dict:
        key = hashlib.sha1(gray.tobytes()).hexdigest()
        cache = st.session_state.setdefault("score_cache", {})
        if key not in cache:
            cache[key] = {k: float(p.predict_proba([gray])[0]) for k, p in self.service.pipelines.items()}
        return cache[key]

    # ========================================================= OVERVIEW
    def page_overview(self):
        title("Casting defect detection under class imbalance",
              "Six classifiers, seven ways of handling imbalance, one test set of 195 unseen images.")
        n_runs = int((self.data.results.tag == "main").sum()) if self.data.results is not None else 0
        k = st.columns(4)
        kpi(k[0], "Images", "1,300", "781 defective · 519 OK")
        kpi(k[1], "Models", "6", "4 classical + 2 neural", C["accent2"])
        kpi(k[2], "Worst imbalance", "20 : 1", "363 OK vs 18 defective", C["accent"])
        kpi(k[3], "Experiment runs", f"{n_runs:,}", "model × ratio × method × repeat", C["ok"])

        tests = self.data.test_images()
        if not tests.empty:
            section("Example test images")
            rs = np.random.RandomState(3)
            rows = pd.concat([tests[tests.label == 0].sample(3, random_state=rs),
                              tests[tests.label == 1].sample(3, random_state=rs)])
            cols = st.columns(6)
            for col, r in zip(cols, rows.itertuples()):
                show_image(col, cv2.imread(r.path, cv2.IMREAD_GRAYSCALE), "OK" if r.label == 0 else "Defective")

        section("Training set at each imbalance ratio")
        rows = [{"Ratio": RATIO_SHORT[r], "Class": c, "Images": n}
                for r in RATIOS
                for c, n in (("OK", N_TRAIN_OK),
                             ("Defective", N_TRAIN_DEF if r == "natural" else round(N_TRAIN_OK / int(r))))]
        df = pd.DataFrame(rows)
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=16).encode(
            y=alt.Y("Class:N", sort=["OK", "Defective"], title=None, axis=None),
            x=alt.X("Images:Q", scale=alt.Scale(domain=[0, 620]), title="Training images"),
            color=alt.Color("Class:N", scale=alt.Scale(domain=["OK", "Defective"], range=[C["ok"], C["bad"]]),
                            legend=alt.Legend(orient="top", title=None)))
        txt = bars.mark_text(align="left", dx=4, fontSize=12, color=C["text"]).encode(text="Images:Q")
        with st.container(border=True):
            show_chart((bars + txt).properties(height=44, width=800).facet(
                row=alt.Row("Ratio:N", sort=[RATIO_SHORT[r] for r in RATIOS], title=None,
                            header=alt.Header(labelAngle=0, labelAlign="left", labelFontWeight="bold")),
                spacing=4))
            st.caption("Only the training set is changed. Validation and test sets keep the natural split.")
        if self.data.summary is not None:
            v = self.data.value
            highlight(f"<b>Best set-up:</b> MobileNetV2 + threshold moving. At 20:1 it reaches "
                      f"{v('MobileNetV2', '20', 'threshold_moving'):.3f} balanced accuracy, against "
                      f"{v('MobileNetV2', '20', 'none'):.3f} with no fix. The SVM + threshold moving matches it at 20:1 "
                      f"({v('SVM', '20', 'threshold_moving'):.3f}) and is the best classical set-up.")

    # ======================================================== INSPECTOR
    def page_inspect(self):
        title("Inspector", "Classify a test image or your own photo with the two deployed models.")
        if not self._models_ready():
            return
        pipes = self.service.pipelines
        thr = {k: float(p.threshold) for k, p in pipes.items()}
        left, mid, right = st.columns([1.05, 1, 1.35])
        with left.container(border=True):
            tab_s, tab_u = st.tabs(["Test samples", "Upload"])
            with tab_u:
                up = st.file_uploader("JPG / PNG / BMP", type=["jpg", "jpeg", "png", "bmp"], key="insp_up")
                st.caption("Best with top-view grayscale impeller photos.")
            with tab_s:
                picked = self.pick_test_image("insp")
        item = self.read_upload(up) if up else picked
        if item is None:
            mid.info("Choose an image to start.")
            return
        name, colour, truth = item
        gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        reasons = self.unfamiliar(colour)
        sc = self.scores(gray)
        verdicts = {k: ("Defective" if sc[k] >= thr[k] else "OK") for k in ("mobilenet", "classical")}
        agree = len(set(verdicts.values())) == 1
        decision = verdicts["mobilenet"] if agree and not reasons else "Needs review"
        with mid.container(border=True):
            show_image(st, gray, name)
            if reasons:
                md('<div class="muted">⚠️ Unfamiliar: ' + "; ".join(reasons) + "</div>")
        with right.container(border=True):
            icon = {"OK": "✅", "Defective": "⛔"}.get(decision, "⚠️")
            text = {"OK": "OK", "Defective": "Defective"}.get(decision, "Needs review")
            md(f'<div class="center" style="margin:.3rem 0 .6rem"><span class="badge {css_class(decision)}">'
               f'{icon} {text}</span></div>')
            g = st.columns(2)
            md(gauge_svg(sc["mobilenet"], thr["mobilenet"], "MobileNetV2"), g[0])
            md(gauge_svg(sc["classical"], thr["classical"], "SVM"), g[1])
            if truth is not None:
                tn = "Defective" if truth else "OK"
                res = "✅" if decision == tn else "⚠️" if decision == "Needs review" else "❌"
                md(f'<div class="center muted" style="margin-top:.4rem">True label: '
                   f'<span class="pill {css_class(tn)}">{tn}</span> {res}</div>')
        st.caption("Score ≥ threshold → Defective. If the two models disagree, or the photo is unlike the "
                   "training images, the result is *Needs review*.")

    # ======================================================== IMBALANCE
    def page_imbalance(self):
        title("Effect of class imbalance", "All six models with no fix: natural training data vs 20:1.")
        if not self._results_ready():
            return
        metric = st.segmented_control("Metric", ["recall_defective", "balanced_accuracy", "accuracy", "roc_auc"],
                                      default="recall_defective", format_func=METRIC_NAME.get,
                                      key="imb_metric") or "recall_defective"
        d = self.data.main
        d = d[(d.method == "none") & d.ratio.isin(["natural", "20"])]
        w = d.pivot(index="model", columns="ratio", values=metric).reset_index()
        w["Model"] = w.model.map(MODEL_NAME)
        order = w.sort_values("20", ascending=False).Model.tolist()
        y = alt.Y("Model:N", sort=order, title=None, axis=model_axis())
        xs = alt.Scale(domain=[-0.08, 1.04])
        xa = alt.Axis(values=[0, .2, .4, .6, .8, 1], format=".1f")
        col = alt.Color("Model:N", scale=model_scale(), legend=None)
        rule = alt.Chart(w).mark_rule(strokeWidth=4, color="#D5DCE4").encode(
            y=y, x=alt.X("natural:Q", title=METRIC_NAME[metric], scale=xs, axis=xa), x2="20:Q")
        p0 = alt.Chart(w).mark_point(size=150, filled=False, strokeWidth=2.5, opacity=1).encode(
            y=y, x=alt.X("natural:Q", scale=xs), color=col,
            tooltip=["Model", alt.Tooltip("natural:Q", format=".3f", title="Natural")])
        p1 = alt.Chart(w).mark_circle(size=170, opacity=1).encode(
            y=y, x=alt.X("20:Q", scale=xs), color=col, tooltip=["Model", alt.Tooltip("20:Q", format=".3f", title="20:1")])
        lab = alt.Chart(w).mark_text(align="right", dx=-12, fontSize=12, fontWeight="bold").encode(
            y=y, x=alt.X("20:Q", scale=xs), text=alt.Text("20:Q", format=".2f"), color=col)
        with st.container(border=True):
            show_chart((rule + p0 + p1 + lab).properties(height=290))
            st.caption("○ natural training data   ● 20:1 training data")

        section("Across all ratios")
        dd = self.data.main
        dd = dd[dd.method == "none"].copy()
        dd["Model"] = dd.model.map(MODEL_NAME)
        dd["Ratio"] = dd.ratio.map(RATIO_SHORT)
        with st.container(border=True):
            show_chart(alt.Chart(dd).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2.5).encode(
                x=alt.X("Ratio:N", sort=[RATIO_SHORT[r] for r in RATIOS], title="Training ratio (OK : defective)",
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y(f"{metric}:Q", title=METRIC_NAME[metric], scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Model:N", scale=model_scale(), legend=alt.Legend(orient="top", title=None)),
                tooltip=["Model", "Ratio", alt.Tooltip(f"{metric}:Q", format=".3f")]).properties(height=300))
        highlight("Defect recall collapses while accuracy and ROC-AUC fall much less: the models still rank "
                  "defects higher, but the 0.5 threshold is in the wrong place.")

    # ============================================================ FIXES
    def page_fixes(self):
        title("Imbalance-handling methods", "Each row is a model, each dot a method. ○ = no fix.")
        if not self._results_ready():
            return
        c1, c2 = st.columns(2)
        with c1:
            ratio = st.segmented_control("Training ratio", ["3", "10", "20"], default="20",
                                         format_func=RATIO_NAME.get, key="fix_ratio") or "20"
        with c2:
            metric = st.segmented_control("Metric", ["balanced_accuracy", "recall_defective", "recall_ok"],
                                          default="balanced_accuracy", format_func=METRIC_NAME.get,
                                          key="fix_metric") or "balanced_accuracy"
        d = self.data.main
        d = d[d.ratio == ratio].copy()
        d["Model"] = d.model.map(MODEL_NAME)
        d["Method"] = d.method.map(METHOD_NAME)
        best = d[d.method != "none"].sort_values(metric, ascending=False).groupby("model").head(1).copy()
        order = [MODEL_NAME[m] for m in best.sort_values(metric, ascending=False).model]
        y = alt.Y("Model:N", sort=order, title=None, axis=model_axis())
        x0 = max(0.0, np.floor((float(d[metric].min()) - 0.03) * 10) / 10)
        ticks = [round(v, 1) for v in np.arange(x0, 1.0001, 0.1)]
        xsc = alt.Scale(domain=[x0, 1.0 + (1.0 - x0) * 0.42])
        span = d.groupby("Model")[metric].agg(["min", "max"]).reset_index()
        rng = alt.Chart(span).mark_rule(strokeWidth=3, color="#E3E8EE").encode(
            y=y, x=alt.X("min:Q", scale=xsc), x2="max:Q")
        dots = alt.Chart(d[d.method != "none"]).mark_circle(size=150, opacity=.9).encode(
            y=y, x=alt.X(f"{metric}:Q", title=METRIC_NAME[metric], scale=xsc,
                         axis=alt.Axis(values=ticks, format=".1f")),
            color=alt.Color("Method:N", scale=alt.Scale(domain=[METHOD_NAME[m] for m in METHODS[1:]],
                                                        range=[METHOD_COLOUR[m] for m in METHODS[1:]]),
                            legend=alt.Legend(orient="top", title=None, columns=3)),
            tooltip=["Model", "Method", alt.Tooltip(f"{metric}:Q", format=".3f")])
        base = alt.Chart(d[d.method == "none"]).mark_point(size=150, strokeWidth=2.5, color=C["grey"],
                                                           filled=False).encode(
            y=y, x=alt.X(f"{metric}:Q", scale=xsc), tooltip=["Model", "Method", alt.Tooltip(f"{metric}:Q", format=".3f")])
        best["label"] = best.apply(lambda r: f"{r[metric]:.2f} {METHOD_NAME[r.method]}", axis=1)
        lab = alt.Chart(best).mark_text(align="left", dx=10, fontSize=11.5, fontWeight="bold",
                                        color=C["text"]).encode(y=y, x=alt.X(f"{metric}:Q", scale=xsc), text="label:N")
        with st.container(border=True):
            show_chart((rng + base + dots + lab).properties(height=330))

        section("Threshold playground")
        avail = [m for m in MODELS if ("main", m, "20", 0, "none") in self.data.probs]
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 1, 1.4])
            model = c1.selectbox("Model", avail, format_func=MODEL_NAME.get, key="tp_model")
            ratio_t = c2.selectbox("Training ratio", RATIOS, index=3, format_func=RATIO_NAME.get, key="tp_ratio")
            p, yt = self.data.probs[("main", model, ratio_t, 0, "none")], self.data.y_test
            thr = c3.slider("Threshold", 0.01, 0.99, 0.50, 0.01, key="tp_thr")
            m = Evaluator.compute(yt, p, thr)
            left, right = st.columns([1.5, 1])
            df = pd.DataFrame({"Defect score": p, "True class": np.where(yt == 1, "Defective", "OK")})
            hist = alt.Chart(df).mark_bar(opacity=.75, binSpacing=0).encode(
                x=alt.X("Defect score:Q", bin=alt.Bin(extent=[0, 1], step=0.025), scale=alt.Scale(domain=[0, 1]),
                        title="Defect score"),
                y=alt.Y("count()", stack=None, title="Test images"),
                color=alt.Color("True class:N", scale=alt.Scale(domain=["OK", "Defective"],
                                                                range=[C["ok"], C["bad"]]),
                                legend=alt.Legend(orient="top", title=None)))
            rule = alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(color=C["text"], strokeWidth=3).encode(
                x=alt.X("t:Q", title="Defect score"))
            show_chart((hist + rule).properties(height=250), left)
            show_chart(confusion_chart(m["tn"], m["fp"], m["fn"], m["tp"], "", 250), right)
            k = st.columns(3)
            kpi(k[0], "Defect recall", f"{m['recall_defective']:.1%}", f"{m['fn']} missed", C["bad"])
            kpi(k[1], "OK recall", f"{m['recall_ok']:.1%}", f"{m['fp']} false alarms", C["ok"])
            kpi(k[2], "Balanced accuracy", f"{m['balanced_accuracy']:.3f}", "", C["accent2"])

    # ========================================================== COMPARE
    def page_compare(self):
        title("Compare", "Pick two set-ups, or see every result at once.")
        if not self._results_ready():
            return
        tabs = st.tabs(["Head-to-head", "Heat map", "Ranking"])
        with tabs[0]:
            self._head_to_head()
        with tabs[1]:
            self._heatmap()
        with tabs[2]:
            self._ranking()

    def _setup_picker(self, col, key, default):
        main = self.data.main
        with col.container(border=True):
            st.markdown(f"**Set-up {key}**")
            a, b, c = st.columns([1, 1.35, 0.75])
            model = a.selectbox("Model", MODELS, index=MODELS.index(default[0]), format_func=MODEL_NAME.get,
                                key=f"h2h_m{key}")
            methods = [m for m in METHODS if m in main[main.model == model].method.unique()]
            method = b.selectbox("Method", methods, index=methods.index(default[1]) if default[1] in methods else 0,
                                 format_func=METHOD_NAME.get, key=f"h2h_f{key}")
            ratio = c.selectbox("Ratio", RATIOS, index=RATIOS.index(default[2]), format_func=RATIO_SHORT.get,
                                key=f"h2h_r{key}")
        return model, ratio, method

    def _head_to_head(self):
        c = st.columns(2)
        a = self._setup_picker(c[0], "A", ("MobileNetV2", "threshold_moving", "20"))   # -> (model, ratio, method)
        b = self._setup_picker(c[1], "B", ("SVM", "threshold_moving", "20"))
        metrics = ["balanced_accuracy", "recall_defective", "recall_ok", "f1_defective", "mcc", "roc_auc"]
        lab = {k: f"{k}: {MODEL_NAME[s[0]]} · {METHOD_NAME[s[2]]} · {RATIO_SHORT[s[1]]}"
               for k, s in (("A", a), ("B", b))}
        df = pd.DataFrame([{"Set-up": lab[k], "Metric": METRIC_NAME[m], "Score": self.data.value(*s, m)}
                           for k, s in (("A", a), ("B", b)) for m in metrics])
        enc = dict(y=alt.Y("Set-up:N", title=None, axis=None),
                   x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1.08]), title=None,
                           axis=alt.Axis(values=[0, .2, .4, .6, .8, 1], format=".1f")))
        bars = alt.Chart().mark_bar(cornerRadiusEnd=4, height=14).encode(
            **enc, color=alt.Color("Set-up:N", scale=alt.Scale(domain=[lab["A"], lab["B"]],
                                                               range=[C["accent"], C["accent2"]]),
                                   legend=alt.Legend(orient="top", title=None, labelLimit=600, direction="vertical")),
            tooltip=["Set-up", "Metric", alt.Tooltip("Score:Q", format=".3f")])
        text = alt.Chart().mark_text(align="left", dx=4, fontSize=11, color=C["text"]).encode(
            **enc, text=alt.Text("Score:Q", format=".3f"))
        with st.container(border=True):
            show_chart(alt.layer(bars, text, data=df).properties(height=36, width=760).facet(
                row=alt.Row("Metric:N", sort=[METRIC_NAME[m] for m in metrics], title=None,
                            header=alt.Header(labelAngle=0, labelAlign="left", labelFontWeight="bold",
                                              labelPadding=4)), spacing=6))
        cm_cols = st.columns(2)
        for col, (k, s_) in zip(cm_cols, (("A", a), ("B", b))):
            mm = self.data.rep0_metrics(*s_)
            if mm:
                with col.container(border=True):
                    show_chart(confusion_chart(mm["tn"], mm["fp"], mm["fn"], mm["tp"], f"Set-up {k}", 250))

    def _heatmap(self):
        main = self.data.main.copy()
        metric = st.segmented_control("Metric", ["balanced_accuracy", "recall_defective", "recall_ok", "mcc"],
                                      default="balanced_accuracy", format_func=METRIC_NAME.get,
                                      key="hm_metric") or "balanced_accuracy"
        short = {"natural": "Nat.", "3": "3:1", "10": "10:1", "20": "20:1"}
        main["Method"] = main.method.map(METHOD_NAME)
        main["Ratio"] = main.ratio.map(short)
        main["Model"] = main.model.map(MODEL_NAME)
        xs = alt.X("Ratio:N", sort=list(short.values()), title=None,
                   axis=alt.Axis(labelAngle=0, labelOverlap=False, orient="top"))
        ys = alt.Y("Method:N", sort=list(METHOD_NAME.values()), title=None, axis=alt.Axis(labelLimit=220))
        heat = alt.Chart().mark_rect(stroke="white", strokeWidth=2, cornerRadius=3).encode(
            x=xs, y=ys, color=alt.Color(f"{metric}:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[.4, 1]),
                                        title="Score"),
            tooltip=["Model", "Method", "Ratio", alt.Tooltip(f"{metric}:Q", format=".3f")])
        text = alt.Chart().mark_text(fontSize=11).encode(
            x=xs, y=ys, text=alt.Text(f"{metric}:Q", format=".2f"),
            color=alt.condition(f"datum['{metric}'] >= 0.94", alt.value("white"), alt.value(C["text"])))
        with st.container(border=True):
            show_chart(alt.layer(heat, text, data=main).properties(width=190, height=200).facet(
                facet=alt.Facet("Model:N", sort=[MODEL_NAME[m] for m in MODELS], title=None), columns=3))

    def _ranking(self):
        main = self.data.main
        imb = main[main.ratio != "natural"].groupby(["model", "method"])[
            ["balanced_accuracy", "recall_defective", "recall_ok", "mcc"]].mean().reset_index()
        rank = imb.sort_values("balanced_accuracy", ascending=False).head(10).copy()
        rank.insert(0, "Rank", range(1, len(rank) + 1))
        rank["Model"] = rank.model.map(MODEL_NAME)
        rank["Method"] = rank.method.map(METHOD_NAME)
        rank = rank.rename(columns={"balanced_accuracy": "Balanced accuracy", "recall_defective": "Defect recall",
                                    "recall_ok": "OK recall", "mcc": "MCC"})
        cfg = {c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.3f")
               for c in ("Balanced accuracy", "Defect recall", "OK recall")}
        cfg["MCC"] = st.column_config.NumberColumn("MCC", format="%.3f")
        st.caption("Top 10, averaged over the imbalanced ratios 3:1, 10:1 and 20:1.")
        show_table(rank[["Rank", "Model", "Method", "Balanced accuracy", "Defect recall", "OK recall", "MCC"]],
                   column_config=cfg, height=38 * (len(rank) + 1))

    # ================================================================ run
    def run(self):
        page = self.sidebar()
        getattr(self, f"page_{page}")()


if __name__ == "__main__":
    CastingDashboard().run()
