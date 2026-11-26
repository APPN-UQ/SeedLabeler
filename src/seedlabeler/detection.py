"""One sample's full detection, shared by LABEL and PROCESS: find seeds, then SAM draws them."""
from . import ml, processing, sam


def run_detection(project, sample, refine=None):
    """Detect with the project's detector (classic or a trained model), then SAM refine if enabled.

    refine: None follows the project's SAM REFINE setting (PROCESS); False
    skips it, for LABEL, which shows the first outlines at once and refines
    them in a second pass. Returns the sample's annotations (also saved, with
    traits written to its CSVs).
    """
    cfg, st = project.config, project.config.settings
    fn = None if cfg.detector == "classic" else ml.model_detector(project.path(cfg.detector), st.model_conf)
    ann = processing.detect(project, sample, fn)
    if (st.sam_refine if refine is None else refine) and ann["seeds"]:
        o, h, v = processing.load_layers(sample)
        ann["seeds"], _ = sam.refine_all(st.sam_model, o, ann["seeds"], h, sample.meta()["res"])
        sample.save_annotations(ann)
        processing.remeasure(project, sample, ann, (o, h, v))
    return ann


def flags(sample, median_count=None):
    """Things worth a second look after detection (short labels)."""
    out = []
    if sample.needs_detection:
        return out
    m = sample.meta()
    n = len(sample.annotations()["seeds"])
    if m.get("depth_coverage", 1) < 0.1:
        out.append("LOW DEPTH")
    if m.get("swapped_fixed"):
        out.append("L/R FIXED")
    if n == 0:
        out.append("NO SEEDS")
    elif median_count and (n < 0.5 * median_count or n > 1.5 * median_count):
        out.append("COUNT?")
    return out
