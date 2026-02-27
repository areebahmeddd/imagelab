export const geometricBlocks = [
  {
    type: "geometric_reflectimage",
    message0: "Reflect image in %1 direction",
    args0: [
      {
        type: "field_dropdown",
        name: "type",
        options: [["X", "X"], ["Y", "Y"], ["Both", "Both"]]
      }
    ],
    previousStatement: null,
    nextStatement: null,
    style: "geometric_style",
    tooltip: "Reflects the current image in the specified direction - Flips the image across the chosen axis. 'X' reflects the image vertically, 'Y' reflects it horizontally, and 'Both' reflects it across both axes. This transformation is useful for creating mirror images or correcting orientation."
  },
  {
    type: "geometric_rotateimage",
    message0: "Rotate image with angle of %1 and rescale by %2",
    args0: [
      { type: "field_angle", name: "angle", angle: 90 },
      { type: "field_number", name: "scale", value: 1, min: 0 }
    ],
    previousStatement: null,
    nextStatement: null,
    style: "geometric_style",
    tooltip: "Rotates the image by the given angle and rescales - Rotates the image by the specified angle in degrees (positive values rotate counter-clockwise) and rescales it by the given factor. This transformation is useful for correcting orientation, creating artistic effects, or preparing images for further analysis."
  },
  {
    type: "geometric_scaleimage",
    message0: "Scale Image by %1 in X axis and by %2 in Y axis",
    args0: [
      { type: "field_number", name: "fx", value: 1, min: 0 },
      { type: "field_number", name: "fy", value: 1, min: 0 }
    ],
    previousStatement: null,
    nextStatement: null,
    style: "geometric_style",
    tooltip: "Scales the image by the given factors - Resizes the image by scaling factors along the X and Y axes. A factor greater than 1 enlarges the image, while a factor between 0 and 1 reduces its size. This transformation is useful for resizing images for display, analysis, or to fit specific dimensions."
  },
  {
    type: "geometric_cropimage",
    message0: "Crop image to coordinates x1 %1 y1 %2 x2 %3 and y2 %4",
    args0: [
      { type: "field_number", name: "x1", value: 0, min: 0 },
      { type: "field_number", name: "y1", value: 0, min: 0 },
      { type: "field_number", name: "x2", value: 0, min: 0 },
      { type: "field_number", name: "y2", value: 0, min: 0 }
    ],
    previousStatement: null,
    nextStatement: null,
    style: "geometric_style",
    tooltip: "Crops the image to the specified coordinates"
  },
  {
    type: "geometric_affineimage",
    message0: "Apply affine transformation %1 Source points %2 P0 x %3 y %4 %5 P1 x %6 y %7 %8 P2 x %9 y %10 %11 Destination points %12 P0 x %13 y %14 %15 P1 x %16 y %17 %18 P2 x %19 y %20",
    args0: [
      { type: "input_dummy" },
      { type: "input_dummy" },
      { type: "field_number", name: "src_x0", value: 0 },
      { type: "field_number", name: "src_y0", value: 0 },
      { type: "input_dummy" },
      { type: "field_number", name: "src_x1", value: 100 },
      { type: "field_number", name: "src_y1", value: 0 },
      { type: "input_dummy" },
      { type: "field_number", name: "src_x2", value: 0 },
      { type: "field_number", name: "src_y2", value: 100 },
      { type: "input_dummy" },
      { type: "input_dummy" },
      { type: "field_number", name: "dst_x0", value: 50 },
      { type: "field_number", name: "dst_y0", value: 100 },
      { type: "input_dummy" },
      { type: "field_number", name: "dst_x1", value: 150 },
      { type: "field_number", name: "dst_y1", value: 100 },
      { type: "input_dummy" },
      { type: "field_number", name: "dst_x2", value: 50 },
      { type: "field_number", name: "dst_y2", value: 200 }
    ],
    previousStatement: null,
    nextStatement: null,
    style: "geometric_style",
    tooltip: "Applies a custom affine transformation defined by mapping three source points (P0, P1, P2) to three destination points. cv2.getAffineTransform() computes the 2x3 matrix from these pairs, enabling shearing, translation, rotation, and scaling. Defaults reproduce a simple +50px, +100px translation."
  }
];
