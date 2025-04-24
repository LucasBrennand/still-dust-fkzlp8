import * as tf from "@tensorflow/tfjs";
import * as qd from "quickdraw.js";

let model;

export async function loadQuickDrawModel() {
  model = await qd.load();
  console.log("QuickDraw model loaded");
  return model;
}

export async function analyzeDrawing(canvasElement) {
  if (!model) await loadQuickDrawModel();

  // Preprocess drawing to match QuickDraw's expected input
  const processedCanvas = preprocessDrawing(canvasElement);

  // Get top 5 guesses
  const predictions = await model.predict(processedCanvas);

  return {
    guesses: predictions.map((p) => p.className),
    timestamp: new Date().toISOString(),
  };
}

function preprocessDrawing(canvas) {
  // 1. Create temporary canvas
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const ctx = tempCanvas.getContext("2d");

  // 2. Convert to 28x28 grayscale
  ctx.drawImage(canvas, 0, 0, 28, 28);

  // 3. Invert colors (QuickDraw expects white lines on black background)
  const imageData = ctx.getImageData(0, 0, 28, 28);
  for (let i = 0; i < imageData.data.length; i += 4) {
    const avg =
      (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
    imageData.data[i] = 255 - avg; // R
    imageData.data[i + 1] = 255 - avg; // G
    imageData.data[i + 2] = 255 - avg; // B
    imageData.data[i + 3] = 255; // Alpha
  }
  ctx.putImageData(imageData, 0, 0);

  return tempCanvas;
}
