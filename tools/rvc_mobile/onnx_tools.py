"""Helpers that turn a torch module into an Android-ready ONNX graph.

Graph convention used by the app (see android/app/src/main/cpp/engine):

* Every model parameter stays an initializer with its *original* state-dict name.
* Parameters are stored externally in a flat ``*.bin`` file (``store_dtype``), at the
  offsets listed in a JSON manifest.  For fp16 storage a ``Cast(to=float)`` node is
  inserted after the initializer, so computation stays fp32 and ONNX Runtime folds the
  cast (and weight norm, transposes, ...) once at session creation.  (fp16-*compute*
  graphs were tried and dropped: ORT's WebGPU EP loses accuracy with them; reduced
  precision is requested from the execution provider or done by the native decoder.)
* Because the bytes in the ``.bin`` are exactly the raw tensors of the source
  checkpoint, the app can build the ``.bin`` itself from an unmodified RVC ``.pth``.
"""

import io
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

OPSET = 17
ALIGN = 64


def torch_export(module: torch.nn.Module, args: Tuple, input_names: List[str], output_names: List[str],
                 dynamic_axes: Dict[str, Dict[int, str]]) -> onnx.ModelProto:
    module.eval()
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            module, args, buf,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
            opset_version=OPSET, do_constant_folding=False, export_params=True,
            keep_initializers_as_inputs=False, dynamo=False,
        )
    return onnx.load_from_string(buf.getvalue())


class DataLayout:
    """Byte layout of a flat parameter file shared by one or more graphs."""

    def __init__(self, data_file: str):
        self.data_file = data_file
        self.entries: Dict[str, dict] = {}
        self.order: List[str] = []
        self.size = 0

    def add(self, name: str, arr: np.ndarray, dtype: str):
        if name in self.entries:
            return self.entries[name]
        npd = {"f16": np.float16, "f32": np.float32}[dtype]
        data = np.ascontiguousarray(arr.astype(npd))
        off = (self.size + ALIGN - 1) // ALIGN * ALIGN
        e = {"name": name, "dtype": dtype, "shape": list(arr.shape), "offset": off, "nbytes": int(data.nbytes)}
        self.entries[name] = e
        self.order.append(name)
        self.size = off + data.nbytes
        e["_data"] = data
        return e

    def write(self, path: str):
        with open(path, "wb") as f:
            for name in self.order:
                e = self.entries[name]
                f.seek(e["offset"])
                f.write(e["_data"].tobytes())
            f.truncate(self.size)

    def manifest(self) -> List[dict]:
        return [{k: v for k, v in self.entries[n].items() if not k.startswith("_")} for n in self.order]


def externalize(model: onnx.ModelProto, params: Dict[str, np.ndarray], layout: DataLayout,
                store_dtype: str = "f16", atol: float = 0.0) -> List[str]:
    """Move parameter initializers to ``layout`` (stored as ``store_dtype``).

    Returns the list of parameter names referenced by this graph.  Raises if an
    initializer named like a parameter does not hold that parameter's value, which
    would mean the exporter transformed it (and the app could not rebuild it).
    """
    g = model.graph
    used = []
    new_inits = []
    casts = []
    rename = {}
    for init in g.initializer:
        if init.name not in params:
            new_inits.append(init)
            continue
        ref = params[init.name]
        val = numpy_helper.to_array(init)
        if val.shape != ref.shape or not np.allclose(val.astype(np.float64), ref.astype(np.float64), atol=atol, rtol=0):
            raise RuntimeError(f"initializer {init.name} differs from source parameter")
        e = layout.add(init.name, ref, store_dtype)
        t = TensorProto()
        t.name = init.name
        t.dims.extend(ref.shape)
        t.data_type = TensorProto.FLOAT16 if store_dtype == "f16" else TensorProto.FLOAT
        t.data_location = TensorProto.EXTERNAL
        for k, v in (("location", layout.data_file), ("offset", str(e["offset"])), ("length", str(e["nbytes"]))):
            kv = t.external_data.add()
            kv.key, kv.value = k, v
        new_inits.append(t)
        used.append(init.name)
        if store_dtype == "f16" and init.data_type == TensorProto.FLOAT:
            cast_out = init.name + "__f32"
            rename[init.name] = cast_out
            casts.append(helper.make_node("Cast", [init.name], [cast_out], to=TensorProto.FLOAT,
                                          name=init.name + "__cast"))
    del g.initializer[:]
    g.initializer.extend(new_inits)
    for n in g.node:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    nodes = casts + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    return used


def check_all_params_used(used: Iterable[str], params: Dict[str, np.ndarray], prefix: str = ""):
    missing = [k for k in params if k.startswith(prefix) and k not in set(used)]
    if missing:
        raise RuntimeError(f"parameters not found as initializers: {missing[:10]} ...")


def save_graph(model: onnx.ModelProto, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # (onnx.checker insists external data files exist; ORT validates on load instead)
    with open(path, "wb") as f:
        f.write(model.SerializeToString())


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
