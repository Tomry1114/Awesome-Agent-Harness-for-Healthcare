"""Public image-tool argument contracts; no task labels or reference answers."""

ZOOM_DESCRIPTION = (
    'Crop a rectangle. param must be a JSON-encoded string "[x1,y1,x2,y2]", '
    'e.g. "[250,250,750,750]", in 0–1000 coordinates; x1<x2 and y1<y2. '
    'Do not use point syntax "x=500 y=500" (SAM2 only). '
    'Live mode adds 10% padding and returns JPEG; replay returns PNG.'
)
_NUMBER = r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'
ZOOM_PARAM = {
    'type': 'string',
    'description': 'JSON string of four bbox numbers in [0,1000], ordered x1,y1,x2,y2. '
                   'Example: "[250,250,750,750]". A point string is invalid.',
    'pattern': r'^\s*\[\s*' + _NUMBER + (r'\s*,\s*' + _NUMBER) * 3 + r'\s*\]\s*$',
}
