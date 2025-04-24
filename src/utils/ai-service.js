import { GoogleGenerativeAI } from "@google/generative-ai";

const genAI = new GoogleGenerativeAI(
  process.env.VITE_GEMINI_API_KEY || "AIzaSyCAD2I7nPvb75cYdBxMQFqeYXGCKzvJ3F4"
);

// Organized word lists by category for better prompting
const WORD_CATEGORIES = {
  animals: ["cat", "dog", "fish", "lion", "owl", "pig", "duck", "ant"],
  objects: ["ball", "hat", "kite", "umbrella", "box", "book", "car", "key"],
  nature: ["tree", "moon", "rain", "sun", "flower", "apple", "grape", "orange"],
  people: ["girl", "queen", "boy", "baby"],
  body: ["nose", "eye", "foot", "hand", "leg", "ear"],
  other: ["house", "ice", "juice", "milk", "pen", "shoe", "table", "window"],
};

async function simplifyImage(imageBase64) {
  // 1. Create canvas element (works in browser environment)
  const img = await loadImage(imageBase64);
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext("2d");

  // 2. Load image and wait for it
  await new Promise((resolve) => {
    img.onload = resolve;
    img.src = imageBase64;
  });

  // 3. Set canvas dimensions
  canvas.width = img.width;
  canvas.height = img.height;

  // 4. Apply preprocessing steps
  ctx.drawImage(img, 0, 0);

  // Convert to grayscale
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
    data[i] = avg; // R
    data[i + 1] = avg; // G
    data[i + 2] = avg; // B
  }
  ctx.putImageData(imageData, 0, 0);

  // Increase contrast
  ctx.filter = "contrast(1.5)";
  ctx.drawImage(canvas, 0, 0);
  ctx.filter = "none";

  // 5. Return as base64
  return canvas.toDataURL("image/png");
}

export async function analyzeDrawing(imageBase64, targetWord = null) {
  try {
    const processedImage = await simplifyImage(imageBase64);
    // 1. Validate and prepare image data
    if (!processedImage?.match(/^data:image\/(png|jpeg);base64,/)) {
      throw new Error("Invalid image format after processing");
    }

    const base64Data = imageBase64.split(",")[1];
    if (!base64Data || base64Data.length < 100) {
      throw new Error("Image data too small");
    }

    // 2. Configure model with stricter parameters
    const model = genAI.getGenerativeModel({
      model: "gemini-1.5-flash",
      safetySettings: [
        { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" },
        { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_NONE" },
      ],
      generationConfig: {
        temperature: 0.05, // Lower for more deterministic responses
        topP: 0.1, // Focus only on high-probability options
        maxOutputTokens: 3, // Force single word responses
      },
    });

    // 3. Create optimized prompt with visual mapping
    const prompt = createPrompt(targetWord);

    // 4. Two-step analysis process
    const initialGuess = await getInitialGuess(model, prompt, base64Data);
    const finalGuess = await verifyGuess(
      model,
      initialGuess,
      base64Data,
      targetWord
    );

    return formatResponse(finalGuess, targetWord);
  } catch (error) {
    console.error("AI Analysis Failed:", error.message);
    return formatResponse(targetWord || "unknown", "error", false);
  }
}

function createPrompt(targetWord) {
  const categoryHints = Object.entries(WORD_CATEGORIES)
    .map(([category, words]) => `${category}: ${words.join(", ")}`)
    .join("\n");

  return `
  Analyze this children's drawing (ages 5-8) and respond with ONLY ONE word from these categories:

  ${categoryHints}

  STRICT RULES:
  1. SHAPE INTERPRETATION:
     - Circle + lines = "sun"
     - Square + triangle = "house"
     - Oval + 4 lines = "dog"
     - Green triangle + brown line = "tree"
     - Circle + stem = "apple"

  2. COLOR GUIDANCE:
     - Yellow = sun, duck
     - Green = tree, leaf
     - Red = apple, rose
     - Blue = water, sky

  3. CHILD DRAWING PATTERNS:
     - Animals: Round head + stick legs
     - Faces: Circle + dot eyes
     - Vehicles: Box + circle wheels
     - Plants: Straight line + oval top

  4. RESPONSE RULES:
     - Only respond with ONE lowercase word from the lists above
     - If unsure between similar options, choose the simpler one
     - Only say "unknown" if completely unrecognizable

  ${
    targetWord
      ? `
  5. CONTEXT CLUES:
     - Related to: ${targetWord} (but don't say this word)
     - Consider: ${getSimilarWords(targetWord, WORD_CATEGORIES).join(", ")}
  `
      : ""
  }

  EXAMPLE INTERPRETATIONS:
  - 🟡 + ☀️ lines = "sun"
  - 🟫⬜ + 🟩🔺 = "tree"
  - ⚪ + 👀 + 👄 = "face"
  - 🟥 + stem = "apple"
  `;
}

async function getInitialGuess(model, prompt, imageData) {
  const result = await model.generateContent({
    contents: [
      {
        role: "user",
        parts: [
          { text: prompt },
          { inlineData: { data: imageData, mimeType: "image/png" } },
        ],
      },
    ],
  });

  const text = (await result.response).text().trim().toLowerCase();
  return text.replace(/[^a-z]/g, "");
}

async function verifyGuess(model, guess, imageData, targetWord) {
  if (guess === "unknown") return guess;

  const confirmationPrompt = `
  Is "${guess}" absolutely correct for this drawing?
  ${targetWord ? `(Target word relates to: ${targetWord})` : ""}
  Consider these aspects:
  1. Does the main shape match?
  2. Are key features present?
  3. Is it the simplest possible match?
  
  Respond ONLY: "yes" or "no"`;

  const result = await model.generateContent({
    contents: [
      {
        role: "user",
        parts: [
          { text: confirmationPrompt },
          { inlineData: { data: imageData, mimeType: "image/png" } },
        ],
      },
    ],
  });

  const confirmation = (await result.response).text().trim().toLowerCase();
  return confirmation === "yes" ? guess : "unknown";
}

function getSimilarWords(targetWord, categories) {
  const targetCategory = Object.entries(categories).find(([_, words]) =>
    words.includes(targetWord)
  )?.[0];

  return targetCategory ? categories[targetCategory] : [];
}

function formatResponse(guess, targetWord, confidence = null) {
  if (!confidence) {
    confidence =
      guess === targetWord
        ? "high"
        : targetWord && guess !== "unknown"
        ? "medium"
        : "low";
  }

  return {
    guess,
    confidence,
    isCorrect: guess === targetWord,
    timestamp: new Date().toISOString(),
  };
}
