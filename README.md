# Wafer Alignment & Extraction Tool

This tool aligns raw physical wafer tiles to a GDSII design file, extracts individual devices at native resolution, and maps labeled defect coordinates to design microns ($\mu\text{m}$).

### Prerequisites
```bash
pip install numpy opencv-python pillow gdstk
```

### Directory Tree
```text
repository-root/
├── tree.txt
├── config.json                            # Active settings
├── wafer_alignment_and_extraction.py      # Main tool
├── semiconductor_design.gds               # Wafer layout
├── batch_wafers.txt                       # Batch run list
└── folder_of_tiles/                       # Raw microscope tile images
    ├── tile_x001_y001.png
    └── ...
```

### Batch Configuration (`batch_wafers.txt`)
Create a file formatted in 4-line groupings:
```text
Wafer_Sample_01:
"folder_of_tiles"
"none"
"none"
```

### Command Line Interface (CLI)

#### **Stage 1: Alignment & Native Crop Extraction**
Run alignment metrology and extract un-rotated device crops:
```bash
python wafer_alignment_and_extraction.py --batch batch_wafers.txt -c
```
*Optional additions:*
*   `--manual` : Launch the interactive overlay UI to align GDS wireframes manually before cropping.
*   `--shave 10` : Crop margin buffer size inside the rotated frame to eliminate edge artifacts.
*   `--out-dir extracted_cells` : Customize the destination directory for crops.

#### **Stage 2: Interactive Defect Labeling & Assembly**
Annotate defects and reassemble the labeled layout:
```bash
python wafer_alignment_and_extraction.py --batch batch_wafers.txt -d -l
```

#### **Interactive Controls:**
*   **`Mouse Drag`**: Draw a bounding box around a defect.
*   **`1` - `5` keys**: Assign a defect class (`1`: blister, `2`: tear, `3`: delamination, `4`: particulate, `5`: hole).
*   **`Space` / `Right Arrow`**: Save the current cell and advance.
*   **`Left Arrow`**: Go back one cell.
*   **`X`**: Toggle exclusion for damaged/empty devices.
*   **`Esc` / `Q`**: Save data to `{wafer_id}_device_defects.json` and export the final annotated layout to `{wafer_id}_stitched_devices.jpg`.
