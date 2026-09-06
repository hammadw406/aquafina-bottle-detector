# Dataset Card — Aquafina Bottle Detection

## Overview
| | |
|---|---|
| **Task** | Single-class object detection |
| **Class** | `aquafina` |
| **Format** | YOLO (`class x_center y_center width height`, normalized) |
| **Annotation tool** | Roboflow |
| **Split** | 70% train / 20% valid / 10% test |

## Collection strategy

**Positives (150–300 images):**
- Own photos: multiple angles (front/back/side/top), lighting conditions (indoor/outdoor/backlit), distances (close-up to far), backgrounds (desk, fridge, store shelf, hand-held), states (cap on/off, condensation, half-empty, crushed).
- Sizes: 500ml and 1L variants included separately and labeled.
- Web-sourced images for extra variety (verify license/usage rights before any public release of the dataset itself).

**Hard negatives (50–100 images):**
- Competing brands: Dasani, Nestlé Pure Life, Kirkland Signature, local/generic bottled water — anything with a similar clear-PET-bottle-with-blue-label silhouette.
- Backgrounds with no bottle at all.
- This is the single most important lever for precision — without hard negatives the model tends to fire on "any water bottle."

## Labeling protocol
- Tight bounding boxes around the full bottle (including cap).
- Partially occluded bottles: box the visible extent only, and only label if >40% of the bottle is visible.
- Multiple bottles in frame → separate box per instance.
- Ambiguous/blurry cases: exclude rather than mislabel.

## Augmentations (applied in Roboflow / at train time)
- Rotation ±15°, brightness ±25%, blur, mosaic, horizontal flip, slight scale jitter.

## Known limitations
- _(fill in after Stage 1 — e.g. "underrepresented: nighttime/low-light shots", "no dented/damaged bottle variants")_

## Version history
| Version | Date | # Images | Notes |
|---|---|---|---|
| v0.1 | _TBD_ | _TBD_ | Initial collection |
