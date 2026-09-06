// ---- Config ----
const MODEL_PATH = "../models/best.onnx"; // update after Stage 5 export
const IMG_SIZE = 640;
const CLASS_NAMES = ["aquafina"];
const IOU_THRESH = 0.45;

// ---- State ----
let session = null;
let confThresh = 0.4;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const video = document.getElementById("video");

// ---- Model loading ----
async function loadModel() {
  try {
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });
    statusEl.textContent = "Model loaded ✓";
  } catch (err) {
    statusEl.textContent = "Failed to load model — check models/best.onnx path";
    console.error(err);
  }
}

// ---- Letterbox resize (matches Ultralytics preprocessing) ----
function letterbox(img, size) {
  const scale = Math.min(size / img.width, size / img.height);
  const nw = Math.round(img.width * scale);
  const nh = Math.round(img.height * scale);
  const padX = Math.floor((size - nw) / 2);
  const padY = Math.floor((size - nh) / 2);

  const off = document.createElement("canvas");
  off.width = size;
  off.height = size;
  const offCtx = off.getContext("2d");
  offCtx.fillStyle = "rgb(114,114,114)";
  offCtx.fillRect(0, 0, size, size);
  offCtx.drawImage(img, padX, padY, nw, nh);

  return { canvas: off, scale, padX, padY };
}

function imageDataToTensor(imageData, size) {
  const { data } = imageData;
  const floatData = new Float32Array(3 * size * size);
  // HWC (RGBA) -> CHW (RGB), normalized 0-1
  for (let i = 0; i < size * size; i++) {
    floatData[i] = data[i * 4] / 255.0;                      // R
    floatData[size * size + i] = data[i * 4 + 1] / 255.0;     // G
    floatData[2 * size * size + i] = data[i * 4 + 2] / 255.0; // B
  }
  return new ort.Tensor("float32", floatData, [1, 3, size, size]);
}

// ---- IoU + NMS ----
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-6);
}

function nms(boxes, scores, iouThresh) {
  const order = scores.map((s, i) => i).sort((i, j) => scores[j] - scores[i]);
  const keep = [];
  const suppressed = new Set();

  for (const i of order) {
    if (suppressed.has(i)) continue;
    keep.push(i);
    for (const j of order) {
      if (j === i || suppressed.has(j)) continue;
      if (iou(boxes[i], boxes[j]) > iouThresh) suppressed.add(j);
    }
  }
  return keep;
}

// ---- Decode YOLOv8 output: [1, 4+num_classes, 8400] ----
function postprocess(output, scale, padX, padY) {
  const dims = output.dims; // [1, 5, 8400] for single class
  const numAttrs = dims[1];
  const numBoxes = dims[2];
  const data = output.data;

  const boxes = [];
  const scores = [];
  const classIds = [];

  for (let i = 0; i < numBoxes; i++) {
    let bestScore = -Infinity;
    let bestClass = 0;
    for (let c = 4; c < numAttrs; c++) {
      const s = data[c * numBoxes + i];
      if (s > bestScore) {
        bestScore = s;
        bestClass = c - 4;
      }
    }
    if (bestScore < confThresh) continue;

    const cx = data[0 * numBoxes + i];
    const cy = data[1 * numBoxes + i];
    const w = data[2 * numBoxes + i];
    const h = data[3 * numBoxes + i];

    let x1 = cx - w / 2;
    let y1 = cy - h / 2;
    let x2 = cx + w / 2;
    let y2 = cy + h / 2;

    // undo letterbox padding/scale to get coords in original image space
    x1 = (x1 - padX) / scale;
    y1 = (y1 - padY) / scale;
    x2 = (x2 - padX) / scale;
    y2 = (y2 - padY) / scale;

    boxes.push([x1, y1, x2, y2]);
    scores.push(bestScore);
    classIds.push(bestClass);
  }

  const keep = nms(boxes, scores, IOU_THRESH);
  return keep.map((i) => ({ box: boxes[i], score: scores[i], classId: classIds[i] }));
}

// ---- Draw ----
function drawDetections(img, detections) {
  canvas.width = img.width;
  canvas.height = img.height;
  ctx.drawImage(img, 0, 0);

  detections.forEach(({ box, score, classId }) => {
    const [x1, y1, x2, y2] = box;
    ctx.strokeStyle = "#ef6a3a";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${CLASS_NAMES[classId]} ${(score * 100).toFixed(0)}%`;
    ctx.font = "16px sans-serif";
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = "#ef6a3a";
    ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 12, 22);
    ctx.fillStyle = "white";
    ctx.fillText(label, x1 + 6, Math.max(14, y1 - 6));
  });

  if (detections.length === 0) {
    statusEl.textContent = "No Aquafina bottle detected";
  } else {
    statusEl.textContent = `Detected ${detections.length} bottle(s)`;
  }
}

// ---- Main inference entrypoint ----
async function runInference(imgElement) {
  if (!session) {
    statusEl.textContent = "Model still loading…";
    return;
  }
  const { canvas: letterboxed, scale, padX, padY } = letterbox(imgElement, IMG_SIZE);
  const lctx = letterboxed.getContext("2d");
  const imageData = lctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const inputTensor = imageDataToTensor(imageData, IMG_SIZE);

  const inputName = session.inputNames[0];
  const results = await session.run({ [inputName]: inputTensor });
  const output = results[session.outputNames[0]];

  const detections = postprocess(output, scale, padX, padY);
  drawDetections(imgElement, detections);
}

// ---- Event wiring ----
document.getElementById("imageInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => runInference(img);
  img.src = URL.createObjectURL(file);
});

document.getElementById("confSlider").addEventListener("input", (e) => {
  confThresh = parseFloat(e.target.value);
  document.getElementById("confVal").textContent = confThresh.toFixed(2);
});

document.getElementById("webcamBtn").addEventListener("click", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  video.play();
  statusEl.textContent = "Webcam live — detecting…";

  const loop = async () => {
    if (video.readyState >= 2) await runInference(video);
    requestAnimationFrame(loop);
  };
  loop();
});

loadModel();
