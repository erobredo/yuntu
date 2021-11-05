"""Simple test for audio object and collection"""
from yuntu.collection.base import Collection
from yuntu.core.audio.audio import Audio

test_col = Collection()
meta = {"path": "./test_data/20210804_055000.WAV",
        "timeexp": 1.0}

score = 0
print("Begining tests:")
print("1. Build Audio without insert.")
Audio(meta)
print("1. SUCCESS!")
score += 1

print("2. Try to insert audio with missing attributes.")
try:
    Audio(meta, insert=True)
    print("2. FAIL!")
except Exception:
    print("2. SUCCESS!")
    score += 1

print("3. Try to insert audio with wrong attributes.")
meta["metadata"] = {"sitio": "Estación de los Tuxtlas", "estado": "Veracruz"}
meta["spectrum"] = "ultrasonic"
try:
    Audio(meta, insert=True)
    print("3. FAIL!")
except Exception:
    print("3. SUCCESS!")
    score += 1

print("4. Inserting Audio with full attributes.")
meta["spectrum"] = "audible"
Audio(meta, insert=True)
print("4. SUCCESS!")
score += 1

print("5. Retrieve media from collection without filters.")
matches = test_col.media(iterate=False)
if len(matches) == 0:
    raise ValueError("Insertion failed!")
print("5. SUCCESS!")
score += 1

print("6. Retrieve media from collection with filter.")
matches = test_col.media(query=lambda c: c.id == 1, iterate=False)
if len(matches) == 0:
    raise ValueError("Insertion failed!")
print("6. SUCCESS!")
score += 1

print("Final score: "+str(score)+" of 6")
