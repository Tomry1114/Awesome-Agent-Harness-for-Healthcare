Complete the supplied medical task using the case facts and any available images. Explain your conclusion, relevant evidence, uncertainty and safety concerns. Do not invent missing findings. Tool use is optional when the supplied evidence suffices.
Images are identified by image_id; img_last refers to the latest available image. The initial observation identifies live image mode when selected. In live mode, zoom and configured segmentation workers return new images. Otherwise segmentation, examination and test tools use recorded exact-request observations. Missing services or results are unavailable evidence, never negative findings. No live patient simulator is connected.
Available tools:
{tools}
Return one <tool_call>{"name":"TOOL", "arguments":{...}}</tool_call> or <answer>your answer</answer>.
