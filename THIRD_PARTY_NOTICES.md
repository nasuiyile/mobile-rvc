# Third-party notices

mobile-rvc is licensed under the MIT License (see `LICENSE`). Parts of it are
ported from, or bundle, the third-party works listed below. Their copyright
notices and license texts are reproduced here as their licenses require.

## Ported source code

| Project | Used for | License |
|---|---|---|
| [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | Inference pipeline, realtime processing, synthesizer structure (`android/app/src/main/cpp/engine`, `tools/rvc_mobile`) | MIT |
| TorchGate by Asaf Zorea, as vendored in RVC `tools/torchgate` | Spectral noise gate (`noise_gate.cpp`) | MIT (via RVC) |
| [CNChTu/FCPE](https://github.com/CNChTu/FCPE) (torchfcpe) | Pitch estimator export and STFT front end | MIT |

## Bundled models

| Model | Source | License |
|---|---|---|
| hubert_base (ContentVec/HuBERT features) | RVC release assets, `huggingface.co/lj1995/VoiceConversionWebUI` | MIT (RVC) |
| FCPE weights | torchfcpe package | MIT |

## Bundled libraries

| Library | License |
|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) (`onnxruntime-android`, `onnxruntime-android-qnn`) | MIT |
| Qualcomm QNN runtime (pulled in by `onnxruntime-android-qnn`, `qnn` variant only) | Qualcomm license terms shipped with that package |
| [Shizuku API](https://github.com/RikkaApps/Shizuku-API) | Apache-2.0 |
| [HiddenApiBypass](https://github.com/LSPosed/AndroidHiddenApiBypass) | Apache-2.0 |
| AndroidX / Jetpack Compose | Apache-2.0 |

The Apache License 2.0 text is available at <https://www.apache.org/licenses/LICENSE-2.0>.

---

## RVC-Project/Retrieval-based-Voice-Conversion-WebUI

```
MIT License

Copyright (c) 2023 liujing04
Copyright (c) 2023 源文雨
Copyright (c) 2023 Ftps

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## CNChTu/FCPE (torchfcpe)

```
MIT License

Copyright (c) 2023 CN_ChiTu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
