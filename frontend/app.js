// ================= CONFIG =================
const API_URL = "http://127.0.0.1:8000/api/verify";

const GESTURES = [
  "palm",
  "l",
  "fist",
  "fist_moved",
  "thumb",
  "index",
  "ok",
  "palm_moved",
  "c",
  "down"
];

let expectedGesture = null;

// ================= DOM =================
const video = document.getElementById("video");
const captureBtn = document.getElementById("captureBtn");
const resultBox = document.getElementById("result");
const challengeText = document.getElementById("challengeText");
const overlayText = document.getElementById("overlayText");

// ================= CAMERA =================
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" }
    });
    video.srcObject = stream;
  } catch (err) {
    alert("❌ Camera access denied");
    console.error(err);
  }
}

startCamera();

// ================= HELPERS =================
function prettify(label) {
  return label.replace("_", " ").toUpperCase();
}

// ================= RANDOM CHALLENGE =================
function generateChallenge() {
  expectedGesture =
    GESTURES[Math.floor(Math.random() * GESTURES.length)];

  challengeText.innerText =
    `👉 Show this gesture: ${prettify(expectedGesture)}`;

  overlayText.innerText = prettify(expectedGesture);
}

generateChallenge();

// ================= CAPTURE & VERIFY =================
captureBtn.addEventListener("click", async () => {
  resultBox.innerHTML = "⏳ Verifying...";

  const canvas = document.createElement("canvas");
  const size = Math.min(video.videoWidth, video.videoHeight);
  canvas.width = 128;
  canvas.height = 128;

  const ctx = canvas.getContext("2d");

  // ---------- MIRROR IMAGE ----------
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  // ---------- CENTER CROP ----------
  const sx = (video.videoWidth - size) / 2;
  const sy = (video.videoHeight - size) / 2;

  ctx.drawImage(
    video,
    sx,
    sy,
    size,
    size,
    0,
    0,
    canvas.width,
    canvas.height
  );

  const blob = await new Promise(resolve =>
    canvas.toBlob(resolve, "image/jpeg")
  );

  const formData = new FormData();
  formData.append("image", blob, "gesture.jpg");

  try {
    const response = await fetch(
      `${API_URL}?expected_gesture=${expectedGesture}`,
      {
        method: "POST",
        body: formData
      }
    );

    const data = await response.json();

    resultBox.innerHTML = `
      <p><b>Expected:</b> ${prettify(data.expected)}</p>
      <p><b>Predicted:</b> ${prettify(data.predicted)}</p>
      <p><b>Confidence:</b> ${(data.confidence * 100).toFixed(2)}%</p>
      <p>
        <b>Status:</b>
        <span style="color:${data.verified ? "lime" : "red"}">
          ${data.verified ? "VERIFIED ✅" : "FAILED ❌"}
        </span>
      </p>
    `;

    // Auto-next challenge on success
    if (data.verified) {
      setTimeout(generateChallenge, 1500);
    }

  } catch (err) {
    console.error(err);
    resultBox.innerHTML = "❌ Verification failed";
  }
});
