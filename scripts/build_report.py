"""Build the Module 4 final report from tracked documentation and Phase 8 records.

The report is deliberately data-driven: empirical values are loaded from the serialized
Phase 8 JSON records rather than retyped into the document.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


REPO_URL = "https://github.com/minnocent12/csc8830-module-4"
DATASET_URL = "https://www.kaggle.com/datasets/aalborguniversity/trimodal-people-segmentation"
AAU_URL = "https://vap.aau.dk/vap-trimodal-people-segmentation-dataset/"
SAM2_URL = "https://github.com/facebookresearch/sam2"
FIXED_FRAMES = ("00085", "00135", "00185")

NAVY = "17365D"
BLUE = "D9EAF7"
LIGHT = "F2F5F8"
GRAY = "667085"
BLACK = "1F2937"


def set_cell_shading(cell: Any, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_border(cell: Any, color: str = "CBD5E1", size: str = "4") -> None:
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element = borders.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_repeat_table_header(row: Any) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_cell_text(cell: Any, text: str, *, bold: bool = False, color: str = BLACK) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run(str(text))
    run.bold = bold
    run.font.name = "Aptos"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> Any:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    header = table.rows[0]
    set_repeat_table_header(header)
    for index, label in enumerate(headers):
        set_cell_text(header.cells[index], label, bold=True, color="FFFFFF")
        set_cell_shading(header.cells[index], NAVY)
    for row_values in rows:
        row = table.add_row()
        for index, value in enumerate(row_values):
            set_cell_text(row.cells[index], value)
            set_cell_shading(row.cells[index], LIGHT if len(table.rows) % 2 == 0 else "FFFFFF")
    for row in table.rows:
        for cell in row.cells:
            set_cell_border(cell)
    if widths:
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    paragraph = doc.add_heading(text, level=level)
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 7)
    paragraph.paragraph_format.space_after = Pt(5)


def add_body(doc: Document, text: str, *, italic: bool = False, color: str = BLACK) -> Any:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.08
    run = paragraph.add_run(text)
    run.italic = italic
    run.font.name = "Aptos"
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor.from_string(color)
    return paragraph


def add_bullet(doc: Document, text: str) -> Any:
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(3)
    run = paragraph.add_run(text)
    run.font.name = "Aptos"
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor.from_string(BLACK)
    return paragraph


def add_caption(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(8)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.name = "Aptos"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor.from_string(GRAY)


def add_hyperlink(paragraph: Any, text: str, url: str) -> None:
    relationship_id = paragraph.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    properties.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.append(underline)
    run.append(properties)
    text_element = OxmlElement("w:t")
    text_element.text = text
    run.append(text_element)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_figure(doc: Document, path: Path, caption: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required evidence figure is missing: {path}")
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    inline_shape = run.add_picture(str(path), width=Inches(5.25))
    inline_shape._inline.docPr.set("title", caption)
    inline_shape._inline.docPr.set("descr", caption)
    add_caption(doc, caption)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.78)
    section.right_margin = Inches(0.78)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    for style_name, size, color in (("Title", 26, NAVY), ("Heading 1", 16, NAVY), ("Heading 2", 12, "245B8F")):
        style = styles[style_name]
        style.font.name = "Aptos Display" if style_name in {"Title", "Heading 1"} else "Aptos"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def load_records(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "results/metrics/phase8_experiment_records.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    expected = {(f"aau-vap-scene1-frame-{frame}", modality) for frame in FIXED_FRAMES for modality in ("rgb", "thermal")}
    actual = {(record["image_id"], record["modality"]) for record in records}
    if len(records) != 6 or actual != expected:
        raise ValueError(f"Unexpected Phase 8 record set: {sorted(actual)}")
    for record in records:
        if record["reference_status"] != "available" or record["reference_type"] != "ground_truth":
            raise ValueError("The final report requires six available ground-truth records")
    return sorted(records, key=lambda item: (item["image_id"], item["modality"]))


def build_report(project_root: Path, output: Path) -> None:
    records = load_records(project_root)
    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("CSc 8830 Computer Vision\nModule 4")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_before = Pt(18)
    run = subtitle.add_run("Human Boundary Detection and Fourier-Domain Analysis")
    run.bold = True
    run.font.name = "Aptos Display"
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor.from_string("245B8F")
    for text in ("Final Assignment Report", "Georgia State University", "Python 3 and OpenCV", "September 2026"):
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(3)
        item = paragraph.add_run(text)
        item.font.name = "Aptos"
        item.font.size = Pt(11)
        item.font.color.rgb = RGBColor.from_string(GRAY)
    add_body(doc, "Evidence scope: implementation and automated tests are complete; the six fixed AAU VAP RGB/thermal cases are experimentally validated only for the recorded subset and procedure. Real SAM2 inference remains pending official runtime setup.", italic=True, color=GRAY)
    doc.add_page_break()

    add_heading(doc, "1. Introduction and Objective")
    add_body(doc, "This report documents an independent Module 4 implementation for classical human-boundary detection in RGB and thermal imagery, a separated SAM2 reference-comparison pathway, and Fourier-domain edge, filtering, derivative, Laplacian, and region-segmentation theory. The implementation is organized around the three assignment questions and exposes the functionality through a standalone Streamlit application.")
    add_body(doc, "The classical RGB pipeline uses an explicitly supplied region of interest and GrabCut-based foreground estimation followed by transparent morphology and component handling. The thermal pipeline converts the decoded three-channel false-color source to intensity, evaluates bright and dark Otsu hypotheses, and records the selected polarity and warnings. Neither pipeline uses a reference mask to generate its prediction.")
    add_table(doc, ["Claim category", "Status in this report"], [
        ["Implemented", "RGB, thermal, evaluation, isolated SAM2 adapter, Fourier helpers, Streamlit pages, and evidence exporter."],
        ["Tested", "Automated regression suite and record/artifact consistency checks."],
        ["Experimentally validated", "Six fixed AAU VAP RGB/thermal cases: three frames × two modalities."],
        ["Pending user data collection", "Real SAM2 inference, demonstration video, and final submission packaging."],
    ], [1.5, 5.7])

    add_heading(doc, "2. Question 1 — RGB Human Boundary Detection")
    add_body(doc, "The RGB workflow validates the decoded OpenCV BGR image, accepts a user ROI, initializes GrabCut with five iterations, converts the result to a canonical boolean mask, and applies the documented 3-pixel opening, 5-pixel closing, and minimum component area of 16 pixels. The final boundary overlay and intermediate masks are available through the RGB page and the tracked evidence artifacts.")
    add_body(doc, "The ROI is an explicit user input and is recorded per case. For Phase 8, the three ROIs were fixed before reference comparison: (120, 20, 450, 460), (80, 80, 350, 400), and (50, 60, 350, 420) for frames 00085, 00135, and 00185 respectively.")
    add_heading(doc, "3. Question 2 — Thermal Human Boundary Detection", level=1)
    add_body(doc, "The thermal workflow treats the downloaded JPG as a three-channel false-color representation rather than calibrated temperature. It converts BGR to intensity, evaluates both bright and dark threshold polarities with Otsu thresholding, applies the documented morphology and component filters, and selects the candidate using the existing polarity-selection score. Phase 8 selected dark polarity on all three cases.")
    add_body(doc, "The thermal records retain the false-color/intensity warning and border-touching-component caution. These warnings are important limitations: the experiment does not support claims about temperature calibration, thermal generalization, or superiority over RGB.")

    add_heading(doc, "4. SAM2 Reference Comparison")
    add_body(doc, "The repository contains an isolated adapter following the official SAM2 image-predictor contract. The Comparison and Evaluation page runs the classical pipeline first, accepts an independent SAM2 box prompt, validates a returned binary mask, and labels it as a SAM2 reference segmentation rather than ground truth. SAM2 cannot alter the classical RGB or thermal pipeline.")
    paragraph = add_body(doc, "Phase 8 verified that the current environment lacks the official SAM2 package, PyTorch, and a local checkpoint. Real SAM2 inference is therefore blocked before inference; there are no SAM2 masks, timings, scores, or metrics in this report. The official implementation is documented at ")
    add_hyperlink(paragraph, "facebookresearch/sam2", SAM2_URL)
    add_body(doc, ". No fabricated SAM2 output or screenshot is included.")

    add_heading(doc, "5. Evaluation Methodology")
    add_body(doc, "Predictions and references are validated as aligned 2D binary masks. The reference source is the AAU VAP nonzero person-label mask canonicalized to foreground 255 and background 0. Shape mismatches fail unless an explicitly recorded nearest-neighbor alignment is justified. All six Phase 8 records have matching 480 × 640 dimensions and no alignment transformation.")
    add_body(doc, "The pixel-level metrics are defined as follows, with TP, FP, FN, and TN taken from the binary confusion matrix:")
    add_table(doc, ["Metric", "Definition and edge convention"], [
        ["IoU", "TP / (TP + FP + FN); 1.0 when both masks have empty foreground."],
        ["Dice", "2TP / (2TP + FP + FN); 1.0 when both masks have empty foreground."],
        ["Precision", "TP / (TP + FP); 1.0 only when both masks are empty, otherwise 0.0 for a zero denominator."],
        ["Recall", "TP / (TP + FN); 1.0 only when both masks are empty, otherwise 0.0 for a zero denominator."],
    ], [1.4, 5.8])

    add_heading(doc, "6. Phase 8 Controlled Real-Data Experiments")
    add_body(doc, "The real-data source is the AAU VAP Trimodal People Segmentation Dataset, version 3, obtained through the public Kaggle distribution. The source also documents the official AAU project page and the CC BY 4.0 terms. The cited dataset paper is Palmero, Clapes, Bahnsen, Mogelmose, Moeslund, and Escalera (2016), Multi-modal RGB-Depth-Thermal Human Body Segmentation, International Journal of Computer Vision, 118(2), 217–239.")
    paragraph = add_body(doc, "Dataset links: ")
    add_hyperlink(paragraph, "Kaggle distribution", DATASET_URL)
    paragraph.add_run("; ")
    add_hyperlink(paragraph, "official AAU project page", AAU_URL)
    add_body(doc, "The selection rule was fixed before metric computation: Scene 1 frame IDs 00085, 00135, and 00185 in ascending order, with both RGB and thermal modalities. No case was selected or removed by score. The machine-readable manifest and serialized records are the authoritative provenance sources.")

    table_rows: list[list[str]] = []
    for record in records:
        frame = record["image_id"].rsplit("-", 1)[-1]
        note = record["selected_polarity"] or "ROI / GrabCut"
        table_rows.append([frame, record["modality"].upper(), "AAU VAP mask", fmt(record["iou"]), fmt(record["dice"]), fmt(record["precision"]), fmt(record["recall"]), note])
    add_table(doc, ["Frame", "Mode", "Reference", "IoU", "Dice", "Precision", "Recall", "Recorded choice"], table_rows, [0.55, 0.55, 1.15, 0.65, 0.65, 0.8, 0.65, 1.6])
    add_caption(doc, "Table 1. Six rows generated directly from results/metrics/phase8_experiment_records.json.")

    means: dict[str, dict[str, float]] = {}
    for modality in ("rgb", "thermal"):
        subset = [record for record in records if record["modality"] == modality]
        means[modality] = {metric: statistics.fmean(record[metric] for record in subset) for metric in ("iou", "dice", "precision", "recall")}
    add_table(doc, ["Modality", "N", "Mean IoU", "Mean Dice", "Mean Precision", "Mean Recall", "Interpretation"], [
        ["RGB", "3", *(fmt(means["rgb"][metric]) for metric in ("iou", "dice", "precision", "recall")), "Descriptive fixed-subset result"],
        ["Thermal", "3", *(fmt(means["thermal"][metric]) for metric in ("iou", "dice", "precision", "recall")), "Descriptive fixed-subset result"],
    ], [0.7, 0.35, 0.7, 0.75, 0.9, 0.75, 2.3])
    add_caption(doc, "Table 2. RGB-versus-thermal comparison is limited to the three paired fixed frames and is not a dataset-wide estimate.")
    add_body(doc, f"Within this fixed N=3 paired subset, RGB IoU values range from {min(record['iou'] for record in records if record['modality'] == 'rgb'):.4f} to {max(record['iou'] for record in records if record['modality'] == 'rgb'):.4f}, while thermal IoU values range from {min(record['iou'] for record in records if record['modality'] == 'thermal'):.4f} to {max(record['iou'] for record in records if record['modality'] == 'thermal'):.4f}. The descriptive means are {means['rgb']['iou']:.4f} for RGB and {means['thermal']['iou']:.4f} for thermal. This is an observation of the stored subset under fixed parameters, not evidence of universal modality superiority.")

    add_heading(doc, "7. Evidence Figures")
    figure_dir = project_root / "results"
    add_figure(doc, figure_dir / "rgb/aau-vap-scene1-00085-rgb_boundary_overlay.png", "Figure 1. RGB boundary overlay for fixed frame 00085.")
    add_figure(doc, figure_dir / "comparisons/aau-vap-scene1-00085-rgb_ground_truth_overlap.png", "Figure 2. RGB prediction/reference overlap for fixed frame 00085.")
    add_figure(doc, figure_dir / "thermal/aau-vap-scene1-00085-thermal_normalized_intensity.png", "Figure 3. Thermal normalized intensity display for fixed frame 00085; the input is false-color, not calibrated temperature.")
    add_figure(doc, figure_dir / "thermal/aau-vap-scene1-00085-thermal_boundary_overlay.png", "Figure 4. Thermal dark-polarity boundary overlay for fixed frame 00085.")
    add_figure(doc, figure_dir / "comparisons/aau-vap-scene1-00085-thermal_ground_truth_overlap.png", "Figure 5. Thermal prediction/reference overlap for fixed frame 00085.")
    add_body(doc, "The five figures above are selected from tracked generated artifacts. The complete per-case inventory remains under results/rgb, results/thermal, results/comparisons, and results/metrics.")

    add_heading(doc, "8. Question 3 — Fourier-Domain Analysis")
    add_heading(doc, "Part A — 2D Fourier representation", level=2)
    add_body(doc, "For a finite M × N image f[m,n], the implementation uses the DFT F[k,l] = ΣₘΣₙ f[m,n] exp(−j2π(km/M + ln/N)). The inverse includes the 1/(MN) normalization. Magnitude describes the strength of each frequency and phase carries positional/alignment information. fftshift changes display layout only; log(1 + |F|) compresses dynamic range for visualization.")
    add_heading(doc, "Part B — why edges contain high frequencies", level=2)
    add_body(doc, "An edge is a rapid spatial intensity change. Reproducing a narrow step requires a broad range of Fourier components, including high frequencies. High frequencies also contain noise, texture, compression artifacts, and border discontinuities, so frequency emphasis is not automatically a semantic boundary detector.")
    add_heading(doc, "Part C — Gaussian high-pass filtering", level=2)
    add_body(doc, "The implementation uses H_HP(u,v) = 1 − H_LP(u,v), with H_LP(u,v) = exp(−(u² + v²)/(2σ²)). The filtered spectrum is H_HP F and the inverse transform is the spatial edge/detail response. The Gaussian transition is smooth and reduces the ringing associated with an ideal hard cutoff.")
    add_heading(doc, "Part D — Fourier derivative property", level=2)
    add_body(doc, "The derivative property is F{∂f/∂x} = j2πuF(u,v) and F{∂f/∂y} = j2πvF(u,v). The frequency multiplier grows with the relevant spatial frequency, emphasizing rapid changes while retaining the signed phase relationship. The code constructs FFT-compatible frequency grids rather than guessing centered indices.")
    add_heading(doc, "Part E — frequency-domain Laplacian", level=2)
    add_body(doc, "The Laplacian transfer function is F{∇²f} = −4π²(u² + v²)F(u,v). Its quadratic frequency growth emphasizes fine detail and edges but also amplifies high-frequency noise. It is an edge/detail response, not a complete human-region segmentation method.")
    add_heading(doc, "Part F — frequency-domain region segmentation", level=2)
    add_body(doc, "A classical frequency-region workflow transforms an image or local window, selects a radial or directional band, measures selected-band energy, optionally reconstructs a spatial response, and applies transparent thresholding or cleanup. A global FFT loses direct spatial localization, so the implementation's local-frequency demonstration uses windowed spectra; small windows improve localization but reduce frequency resolution, while large windows do the opposite. The Streamlit example is synthetic educational material, not an empirical result.")

    add_heading(doc, "9. Limitations and Reproducibility")
    add_body(doc, "The experiment is limited to six records from three preselected Scene 1 frames, one fixed ROI per modality/case, and the existing classical parameters. The thermal source is false-color JPG data and does not provide calibrated temperature. The result table does not establish robustness, cross-scene performance, or modality superiority. SAM2 inference, video recording, and final Classroom submission remain pending.")
    add_body(doc, "To reproduce the tracked evidence, create the documented Python environment, run python scripts/run_phase8_evidence.py --manifest data/experiment_manifest.json --project-root . --output-dir results, and run python -m pytest -q. The exporter sets the recorded per-case RNG seeds, writes derived artifacts, and rechecks serialized metrics and confusion counts. Do not add ignored raw inputs or model checkpoints to the repository.")
    add_body(doc, "The Streamlit application is launched with streamlit run app.py. Its standalone pages expose RGB Human Boundary, Thermal Human Boundary, Comparison and Evaluation, and Fourier Theory. The report builder itself is scripts/build_report.py and reads the tracked JSON records rather than embedding manually entered empirical values.")

    add_heading(doc, "10. Conclusion")
    add_body(doc, "Module 4 provides a tested classical RGB/thermal workflow, strict reference evaluation, an isolated official-SAM2 integration point, and an explanatory Fourier Parts A–F implementation. The controlled real-data evidence is traceable and limited to the stated six-case subset. The evidence supports the recorded observation that the RGB pipeline produced higher descriptive metrics than the thermal pipeline on these paired cases, while the thermal warnings and SAM2 runtime block remain explicit limitations.")
    add_heading(doc, "Repository")
    paragraph = add_body(doc, "Source code, tests, evidence records, and reproducibility instructions: ")
    add_hyperlink(paragraph, "Module 4 GitHub repository", REPO_URL)
    add_body(doc, "This report does not claim real SAM2 results and does not substitute automated tests for experimental validation.", italic=True, color=GRAY)

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("deliverables/Module_4_Final_Report.docx"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    build_report(root, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
