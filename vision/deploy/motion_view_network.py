#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import onnx
import torch
from onnx import compose

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion_view import CONTEXT_BRIGHTNESS, HIGHLIGHT_GREEN_SHARE, MOTION_BOOST

IMAGE_SIZE = 640
GRAY_STACK_INPUT = "grays"
YOLO_INPUT = "images"
OPSET = 11
BLUR_KERNEL = np.outer([1, 2, 1], [1, 2, 1]) / 16.0


def fixed_conv(weights, bias=None, groups=1, padding=0):
    weights = torch.tensor(np.asarray(weights, np.float32))
    conv = torch.nn.Conv2d(weights.shape[1] * groups, weights.shape[0], weights.shape[2], padding=padding, groups=groups, bias=bias is not None)
    conv.weight.data = weights
    if bias is not None:
        conv.bias.data = torch.tensor(np.asarray(bias, np.float32))
    conv.requires_grad_(False)
    return conv


def one_by_one(rows, bias=None):
    return fixed_conv(np.asarray(rows, np.float32)[:, :, None, None], bias)


class MotionViewLayers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blur = fixed_conv(np.stack([BLUR_KERNEL[None]] * 3), groups=3, padding=1)
        self.signed_differences = one_by_one([[-1, 1, 0], [1, -1, 0], [0, 1, -1], [0, -1, 1]])
        self.absolute_differences = one_by_one([[1, 1, 0, 0], [0, 0, 1, 1]])
        self.first_minus_second = one_by_one([[1, -1]])
        self.minimum = one_by_one([[1, 0, -1]])
        self.over_white = one_by_one([[MOTION_BOOST]], bias=[-255.0])
        self.highlight = one_by_one([[MOTION_BOOST, -1]])
        self.context = one_by_one([[0, CONTEXT_BRIGHTNESS, 0]])
        self.above_context = one_by_one([[1, -1], [HIGHLIGHT_GREEN_SHARE, -1]])
        self.rgb_for_yolo = one_by_one([[1 / 255, 1 / 255, 0], [1 / 255, 0, 1 / 255], [1 / 255, 0, 0]])
        self.relu = torch.nn.ReLU()

    def forward(self, grays):
        smooth = self.blur(grays)
        absolute = self.absolute_differences(self.relu(self.signed_differences(smooth)))
        excess = self.relu(self.first_minus_second(absolute))
        moved_now = self.minimum(torch.cat([absolute, excess], 1))
        clipped = self.relu(self.over_white(moved_now))
        highlight = self.highlight(torch.cat([moved_now, clipped], 1))
        context = self.context(grays)
        lifts = self.relu(self.above_context(torch.cat([highlight, context], 1)))
        return self.rgb_for_yolo(torch.cat([context, lifts], 1))


def motion_view_from_layers(previous, current, upcoming):
    stack = np.stack([previous, current, upcoming])[None].astype(np.float32)
    with torch.no_grad():
        rgb = MotionViewLayers()(torch.from_numpy(stack))[0].numpy() * 255
    return np.clip(np.rint(rgb[::-1].transpose(1, 2, 0)), 0, 255).astype(np.uint8)


def export_motion_view_onnx(path, image_size=IMAGE_SIZE):
    torch.onnx.export(MotionViewLayers().eval(), torch.zeros(1, 3, image_size, image_size), path, opset_version=OPSET,
                      input_names=[GRAY_STACK_INPUT], output_names=["motion_view_rgb"], dynamo=False)


def yolo_image_size(yolo):
    return yolo.graph.input[0].type.tensor_type.shape.dim[-1].dim_value


def gray_stack_model(motion_view_onnx, yolo_onnx, output):
    front, yolo = onnx.load(motion_view_onnx), onnx.load(yolo_onnx)
    if yolo_image_size(yolo) != yolo_image_size(front):
        raise SystemExit(f"{yolo_onnx} takes {yolo_image_size(yolo)} x {yolo_image_size(yolo)} images; export it at {yolo_image_size(front)} or pass --image-size {yolo_image_size(yolo)}")
    front.ir_version = yolo.ir_version
    front = compose.add_prefix(front, prefix="motion_view/")
    merged = compose.merge_models(front, yolo, io_map=[("motion_view/motion_view_rgb", YOLO_INPUT)])
    merged.graph.input[0].name = GRAY_STACK_INPUT
    for node in merged.graph.node:
        node.input[:] = [GRAY_STACK_INPUT if name == "motion_view/" + GRAY_STACK_INPUT else name for name in node.input]
    onnx.checker.check_model(merged)
    onnx.save(merged, output)


def main():
    parser = argparse.ArgumentParser(description="Put the motion view in front of YOLO as fixed network layers: the model then takes three gray crops (t-1, t, t+1) instead of the orange motion view.")
    parser.add_argument("--yolo-onnx", default="models/bird_yolo11n_v1.onnx", help="YOLO exported to ONNX at --image-size, opset 11")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="crop side in pixels; 320 for the crop used while following the bird")
    parser.add_argument("--output", default="models/bird_yolo11n_v1_gray_stack.onnx")
    args = parser.parse_args()
    motion_view_onnx = args.output[:-5] + "_front.onnx"
    export_motion_view_onnx(motion_view_onnx, args.image_size)
    gray_stack_model(motion_view_onnx, args.yolo_onnx, args.output)
    os.remove(motion_view_onnx)
    print("saved", args.output)


if __name__ == "__main__":
    main()
