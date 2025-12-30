import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ml.infer import predict_gesture
from app.ml.landmark_verify import verify_by_landmarks

router = APIRouter()

# ================= TEMP UPLOAD DIR =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= ALLOWED VARIANTS =================
ALLOWED_VARIANTS = {
    "ok": ["ok", "l"],
    "l": ["l"],
    "palm": ["palm", "palm_moved"],
    "palm_moved": ["palm", "palm_moved"],
    "fist": ["fist", "fist_moved"],
    "fist_moved": ["fist", "fist_moved"],
    "thumb": ["thumb"],
    "index": ["index"],
    "c": ["c"],
    "down": ["down"]
}

# ================= API =================
@router.post("/verify")
async def verify_gesture(
    expected_gesture: str,
    image: UploadFile = File(...)
):
    # ---------- Validate ----------
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image"
        )

    expected_gesture = expected_gesture.lower()

    # ---------- Save temp file ----------
    filename = f"{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(await image.read())

    try:
        # ---------- CNN prediction ----------
        result = predict_gesture(file_path)
        predicted = result["gesture"].lower()
        confidence = result["confidence"]

        # ---------- CNN verification ----------
        cnn_verified = (
            confidence >= 0.70 and
            predicted in ALLOWED_VARIANTS.get(expected_gesture, [])
        )

        # ---------- Landmark verification ----------
        landmark_verified = verify_by_landmarks(
            file_path,
            expected_gesture
        )

        verified = cnn_verified and landmark_verified

        return {
            "expected": expected_gesture,
            "predicted": predicted,
            "confidence": confidence,
            "verified": verified,
            "cnn_verified": cnn_verified,
            "landmark_verified": landmark_verified
        }

    finally:
        # ---------- Cleanup ----------
        if os.path.exists(file_path):
            os.remove(file_path)
